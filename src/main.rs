use scrap::{Capturer, Display};
use serialport;
use serde::Deserialize;
use std::io::Write;
use std::thread;
use std::time::{Duration, Instant};
use rayon::prelude::*;

#[derive(Debug, Deserialize)]
struct AmbilightConfig {
    fps: u32,
    port_name: String,
    baud_rate: u32,

    top_led_count: usize,
    left_led_count: usize,
    right_led_count: usize,
    bottom_left_led_count: usize,
    bottom_right_led_count: usize,
    offset: usize,

    invert_direction: bool,
    pixel_thickness: usize,

    brightness: usize,
    white_balance_temperature: f32,
    gamma: f32,
}

// Определяем область экрана для одного светодиода
struct LedRegion {
    x1: usize,
    y1: usize,
    x2: usize,
    y2: usize,
}

/// Вычисляет множители для коррекции белого по заданной цветовой температуре (в Кельвинах).
fn color_temperature_to_rgb_multipliers(temp: f32) -> (f32, f32, f32) {
    let temp = temp / 100.0;
    let (r, g, b): (f32, f32, f32);
    if temp <= 66.0 {
        r = 255.0;
        g = 99.4708025861 * (temp.max(1.0)).ln() - 161.1195681661;
        b = if temp <= 19.0 {
            0.0
        } else {
            138.5177312231 * ((temp - 10.0).max(1.0)).ln() - 305.0447927307
        };
    } else {
        r = 329.698727446 * ((temp - 60.0).max(1.0)).powf(-0.1332047592);
        g = 288.1221695283 * ((temp - 60.0).max(1.0)).powf(-0.0755148492);
        b = 255.0;
    }
    (r / 255.0, g / 255.0, b / 255.0)
}

/// Создаёт вектор регионов (LedRegion) в нужном порядке.
fn create_led_regions(config: &AmbilightConfig, width: usize, height: usize) -> Vec<LedRegion> {
    let mut regions = Vec::new();

    let pixel_thickness = height * config.pixel_thickness / 100;

    let total_bottom = config.bottom_left_led_count + config.bottom_right_led_count;
    if total_bottom == 0 {
        return regions; // Если снизу нет диодов, вернём пустой вектор
    }

    let offset_pixels = width * config.offset / 100;
    let effective_width = width.saturating_sub(offset_pixels);
    let left_ratio = config.bottom_left_led_count as f32 / total_bottom as f32;
    let left_group_width = (left_ratio * effective_width as f32).round() as usize;
    let right_ratio = config.bottom_right_led_count as f32 / total_bottom as f32;
    let right_group_width = (right_ratio * effective_width as f32).round() as usize;
    let right_group_start = left_group_width + offset_pixels;

    // 1) Нижняя правая группа: слева → направо
    if config.bottom_right_led_count > 0 {
        let seg_w = right_group_width as f32 / config.bottom_right_led_count as f32;
        for i in 0..config.bottom_right_led_count {
            let x1 = (right_group_start as f32 + i as f32 * seg_w).round() as usize;
            let x2 = (right_group_start as f32 + (i + 1) as f32 * seg_w).round() as usize;
            regions.push(LedRegion {
                x1: x1.min(width),
                y1: height.saturating_sub(pixel_thickness),
                x2: x2.min(width),
                y2: height,
            });
        }
    }

    // 2) Правая сторона: снизу → вверх
    if config.right_led_count > 0 {
        let seg_h = height as f32 / config.right_led_count as f32;
        for i in 0..config.right_led_count {
            let y1 = (height as f32 - (i + 1) as f32 * seg_h).round() as usize;
            let y2 = (height as f32 - i as f32 * seg_h).round() as usize;
            regions.push(LedRegion {
                x1: width.saturating_sub(pixel_thickness),
                y1: y1.min(height),
                x2: width,
                y2: y2.min(height),
            });
        }
    }

    // 3) Верхняя сторона: справа → налево
    if config.top_led_count > 0 {
        let seg_w = width as f32 / config.top_led_count as f32;
        for i in 0..config.top_led_count {
            let rev_i = config.top_led_count - 1 - i;
            let x1 = (rev_i as f32 * seg_w).round() as usize;
            let x2 = ((rev_i + 1) as f32 * seg_w).round() as usize;
            regions.push(LedRegion {
                x1: x1.min(width),
                y1: 0,
                x2: x2.min(width),
                y2: pixel_thickness,
            });
        }
    }

    // 4) Левая сторона: сверху → вниз
    if config.left_led_count > 0 {
        let seg_h = height as f32 / config.left_led_count as f32;
        for i in 0..config.left_led_count {
            let y1 = (i as f32 * seg_h).round() as usize;
            let y2 = ((i + 1) as f32 * seg_h).round() as usize;
            regions.push(LedRegion {
                x1: 0,
                y1: y1.min(height),
                x2: pixel_thickness,
                y2: y2.min(height),
            });
        }
    }

    // 5) Нижняя левая группа: слева → направо
    if config.bottom_left_led_count > 0 {
        let seg_w = left_group_width as f32 / config.bottom_left_led_count as f32;
        for i in 0..config.bottom_left_led_count {
            let x1 = (i as f32 * seg_w).round() as usize;
            let x2 = ((i + 1) as f32 * seg_w).round() as usize;
            regions.push(LedRegion {
                x1: x1.min(width),
                y1: height.saturating_sub(pixel_thickness),
                x2: x2.min(width),
                y2: height,
            });
        }
    }

    regions
}

/// Предварительный расчёт индексов для каждого региона.
/// Для каждого пикселя в регионе вычисляем смещение в буфере кадра.
/// Каждый пиксель занимает 4 байта (BGRA).
fn precompute_region_indices(regions: &[LedRegion], width: usize) -> Vec<Vec<usize>> {
    regions
        .iter()
        .map(|region| {
            let mut indices = Vec::new();
            for y in region.y1..region.y2 {
                // Вычисляем базовое смещение для строки
                let row_base = y * width * 4;
                for x in region.x1..region.x2 {
                    indices.push(row_base + x * 4);
                }
            }
            indices
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Чтение настроек
    let config_data = std::fs::read_to_string("config.toml")?;
    let config: AmbilightConfig = toml::from_str(&config_data)?;
    println!("Настройки: {:#?}", config);

    // 2. Настройка захвата экрана
    let display = Display::primary()?;
    let mut capturer = Capturer::new(display)?;
    let (width, height) = (capturer.width(), capturer.height());
    println!("Экран: {}x{}", width, height);

    // 3. Открытие последовательного порта для Arduino
    let mut port = serialport::new(&config.port_name, config.baud_rate)
        .timeout(Duration::from_millis(10))
        .open()
        .expect("Не удалось открыть порт");

    // 4. Генерация регионов и их оптимизация
    let mut led_regions = create_led_regions(&config, width, height);
    if config.invert_direction {
        led_regions.reverse();
    }
    // Предварительный расчёт смещений (индексов) для каждого региона
    let precomputed_indices = precompute_region_indices(&led_regions, width);

    // Предвычисление множителей для баланса белого
    let (r_mult, g_mult, b_mult) =
        color_temperature_to_rgb_multipliers(config.white_balance_temperature);

    // Предвычисление яркости
    let brightness = (config.brightness as f32) / 100.0;

    // Выделение буфера для формирования пакета Adalight
    let mut msg_buffer = Vec::with_capacity(3 + 3 * precomputed_indices.len());

    // Счётчик FPS
    let mut frame_count = 0;
    let mut fps_timer = Instant::now();

    // Заданная длительность кадра
    let frame_duration = Duration::from_millis(1000 / config.fps as u64);

    'main_loop: loop {
        // Фиксируем время начала обработки кадра (включая ожидание нового кадра)
        let frame_start = Instant::now();

        // 5. Захват кадра: ждем, пока кадр не станет доступным
        let frame = loop {
            match capturer.frame() {
                Ok(frame) => break frame,
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    // Короткий sleep, чтобы не грузить процессор
                    thread::sleep(Duration::from_millis(1));
                },
                Err(e) => {
                    eprintln!("Ошибка захвата: {}", e);
                    thread::sleep(frame_duration);
                    continue 'main_loop;
                }
            }
        };

        frame_count += 1;
        if fps_timer.elapsed() >= Duration::from_secs(1) {
            println!("FPS захвата экрана: {}", frame_count);
            frame_count = 0;
            fps_timer = Instant::now();
        }

        // 6. Параллельный расчёт среднего цвета по регионам
        let colors: Vec<(u8, u8, u8)> = precomputed_indices.par_iter().map(|indices| {
            let mut sum_r: u64 = 0;
            let mut sum_g: u64 = 0;
            let mut sum_b: u64 = 0;
            let count = indices.len() as u64;
            unsafe {
                let ptr = frame.as_ptr();
                for &offset in indices {
                    // Чтение байтов пикселя (порядок: B, G, R, A)
                    let b = *ptr.add(offset);
                    let g = *ptr.add(offset + 1);
                    let r = *ptr.add(offset + 2);
                    sum_r += r as u64;
                    sum_g += g as u64;
                    sum_b += b as u64;
                }
            }
            if count == 0 {
                return (0, 0, 0);
            }

            let avg_r = (sum_r / count) as u8;
            let avg_g = (sum_g / count) as u8;
            let avg_b = (sum_b / count) as u8;

            // Применяем гамма-коррекцию
            let mut r = 255.0 * ((avg_r as f32 / 255.0).powf(config.gamma));
            let mut g = 255.0 * ((avg_g as f32 / 255.0).powf(config.gamma));
            let mut b = 255.0 * ((avg_b as f32 / 255.0).powf(config.gamma));

            // Применяем баланс белого
            r = r * r_mult;
            g = g * g_mult;
            b = b * b_mult;

            // Применяем яркость
            r = (r * brightness).min(255.0);
            g = (g * brightness).min(255.0);
            b = (b * brightness).min(255.0);

            (r as u8, g as u8, b as u8)
        }).collect();

        // 7. Формирование пакета Adalight
        msg_buffer.clear();
        msg_buffer.extend_from_slice(b"Ada");
        let n = colors.len() * 3;
        let hi = (n >> 8) as u8;
        let lo = (n & 0xFF) as u8;
        let chk = hi ^ lo ^ 0x55;
        msg_buffer.extend_from_slice(&[hi, lo, chk]);
        for &(r, g, b) in &colors {
            msg_buffer.push(r);
            msg_buffer.push(g);
            msg_buffer.push(b);
        }

        // let start_timer = Instant::now();
        if let Err(e) = port.write_all(&msg_buffer) {
            eprintln!("Ошибка отправки: {}", e);
        }

        // 8. Вычисляем общее время, затраченное на получение и обработку кадра,
        // и ждём остаток до завершения заданного периода кадра.
        let elapsed = frame_start.elapsed();
        if elapsed < frame_duration {
            thread::sleep(frame_duration - elapsed);
        }
    }
}
