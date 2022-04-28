const fn build_color(r: u8, g: u8, b: u8) -> u32 {
    u32::from_be_bytes([r, g, b, 0xFF])
}

pub const CL_WHITE: u32 = build_color(255, 255, 255);
pub const CL_BLACK: u32 = build_color(0, 0, 0);

pub const CL_RED: u32 = build_color(255, 0, 0);
pub const CL_GREEN: u32 = build_color(0, 255, 0);
pub const CL_BLUE: u32 = build_color(0, 0, 255);

pub const CL_YELLOW: u32 = build_color(255, 255, 0);
pub const CL_CYAN: u32 = build_color(0, 255, 255);
pub const CL_PINK: u32 = build_color(255, 0, 255);

pub const CL_ORANGE: u32 = build_color(255, 122, 0);
pub const CL_PURPLE: u32 = build_color(170, 0, 255);
