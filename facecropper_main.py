if __name__ == "__main__":
    from config import RUN_TEST

    if RUN_TEST:
        from test import run_test
        print("[MAIN] Running test")
        run_test()
    else:
        from module.zm_entry_layer import run_module
        print("[MAIN] Starting FaceCropper module")
        run_module()
