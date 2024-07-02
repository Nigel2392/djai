from typing import Any
from django.conf import settings
from django.utils.module_loading import import_string


class LazyBackend:
    def __init__(self, klass, options: dict):
        self.klass = klass
        self.options = options
        self._backend = None

    def get_backend(self):
        if self._backend is None:
            self._backend = self.klass(self.options)
        return self._backend


class LazyBackendDict(dict):
    def __getitem__(self, key):
        item = super().__getitem__(key)
        if isinstance(item, LazyBackend):
            return item.get_backend()
        return item
    
    def get(self, key, default=None, get_lazy=False):
        try:
            item = super().__getitem__(key)
            if get_lazy:
                return item
            if isinstance(item, LazyBackend):
                return item.get_backend()
            return item
        except KeyError:
            return default


BACKENDS = getattr(settings, "DJAI_BACKENDS", {
    
})
backends = LazyBackendDict()

for backend_name, config in BACKENDS.items():
    klass = config["CLASS"]
    options = config.get("OPTIONS", {})

    if isinstance(klass, str):
        klass = import_string(klass)
    
    backends[backend_name] = LazyBackend(klass, options)


def initialize():
    for backend in backends.values():
        backend.get_backend()


def get_backend(backend_name: str, fail_silently: bool = True, get_lazy: bool = False) -> Any:

    try:
        return backends.get(backend_name, get_lazy=get_lazy)
    except KeyError:
        if fail_silently:
            return None
        raise KeyError(f"Backend {backend_name} not found")

