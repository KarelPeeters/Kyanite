fn main() {
    windows::build!(
        Windows::Win32::WindowsProgramming::PROCESS_CREATION_FLAGS,
        Windows::Win32::SystemServices::{GetCurrentProcess, SetPriorityClass},
    )
}
