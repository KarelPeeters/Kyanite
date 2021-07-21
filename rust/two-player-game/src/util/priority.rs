#[cfg(windows)]
mod bindings {
    ::windows::include_bindings!();
}

#[cfg(windows)]
pub fn lower_process_priority() {
    //TODO redo this once the windows crate is fixed

    // use bindings::Windows::Win32::System::Threading::IDLE_PRIORITY_CLASS;
    use bindings::Windows::Win32::System::Threading::PROCESS_CREATION_FLAGS;
    use bindings::Windows::Win32::System::Threading::{GetCurrentProcess, SetPriorityClass};

    const IDLE_PRIORITY_CLASS: u32 = 0x00000040;

    unsafe {
        SetPriorityClass(GetCurrentProcess(), PROCESS_CREATION_FLAGS(IDLE_PRIORITY_CLASS));
    }
}

#[cfg(not(windows))]
pub fn lower_process_priority() {}