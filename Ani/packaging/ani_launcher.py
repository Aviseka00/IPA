"""Entry point for the frozen (PyInstaller) build."""
import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from ani.__main__ import main
    main()
