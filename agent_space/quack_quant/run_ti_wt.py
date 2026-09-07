import runpy, sys, os
sys.modules["torchvision"] = None
sys.argv = ["test/inductor/test_torchinductor.py"]
runpy.run_path(os.path.join(os.environ["PYTORCH_WORKTREE"], "test/inductor/test_torchinductor.py"), run_name="__main__")
