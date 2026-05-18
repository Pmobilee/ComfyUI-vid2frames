try:
    from .frame_selector import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent / "frame_selector.py"
    spec = importlib.util.spec_from_file_location("aia_frame_selector", path)
    frame_selector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frame_selector)

    NODE_CLASS_MAPPINGS = frame_selector.NODE_CLASS_MAPPINGS
    NODE_DISPLAY_NAME_MAPPINGS = frame_selector.NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
