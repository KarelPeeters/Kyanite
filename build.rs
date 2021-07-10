fn main() {
    #[cfg(windows)]
    windows::build!(
        // Windows::Win32::System::Threading::IDLE_PRIORITY_CLASS,
        Windows::Win32::System::Threading::PROCESS_CREATION_FLAGS,
        Windows::Win32::System::Threading::{GetCurrentProcess, SetPriorityClass},
    )
}
