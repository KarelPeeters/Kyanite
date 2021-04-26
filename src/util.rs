#[cfg(windows)]
mod bindings {
    ::windows::include_bindings!();
}

#[cfg(windows)]
pub fn lower_process_priority() {
    use bindings::Windows::Win32::SystemServices::*;

    unsafe {
        SetPriorityClass(GetCurrentProcess(), PROCESS_PRIORITY_CLASS::IDLE_PRIORITY_CLASS);
    }
}

#[cfg(not(windows))]
pub fn lower_process_priority() {}
