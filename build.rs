fn main() {
    windows::build!(
        Windows::Win32::SystemServices::{GetCurrentProcess, SetPriorityClass, PROCESS_PRIORITY_CLASS},
    )
}
