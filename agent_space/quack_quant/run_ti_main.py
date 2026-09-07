import runpy, sys
sys.modules["torchvision"] = None
sys.argv = ["test/inductor/test_torchinductor.py"]
runpy.run_path("test/inductor/test_torchinductor.py", run_name="__main__")
