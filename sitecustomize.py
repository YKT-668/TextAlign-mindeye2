# auto-loaded by Python if on PYTHONPATH
try:
    import huggingface_hub as hfh
    if not hasattr(hfh, "cached_download"):
        from huggingface_hub import hf_hub_download
        def cached_download(*args, **kw):
            return hf_hub_download(*args, **kw)
        hfh.cached_download = cached_download
except Exception:
    pass
