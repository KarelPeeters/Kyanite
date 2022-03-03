use std::collections::HashMap;
use std::ops::Index;

#[derive(Debug)]
pub struct Store<'a, T> {
    inner: HashMap<&'a str, T>,
}

impl<'a, T> Store<'a, T> {
    pub fn contains(&self, key: &str) -> bool {
        self.inner.contains_key(&key)
    }

    pub fn define(&mut self, key: &'a str, value: T) {
        let prev = self.inner.insert(key, value);
        assert!(prev.is_none(), "Key {} already has a value", key);
    }

    pub fn get(&'a self, key: &'a str) -> Option<&'a T> {
        self.inner.get(&key)
    }
}

impl<'a, T> Index<&str> for Store<'a, T> {
    type Output = T;

    fn index(&self, key: &str) -> &T {
        self.inner.get(key).unwrap_or_else(|| panic!("Key {} not found", key))
    }
}

impl<'a, T> Default for Store<'a, T> {
    fn default() -> Self {
        Store {
            inner: Default::default(),
        }
    }
}
