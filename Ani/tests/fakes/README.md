Stand-ins for optional AI packages (cellpose 3, micro_sam, ultralytics) so the adapters can be tested
without downloading model weights. Function signatures copy the real libraries, so passing a wrong
argument fails here exactly as it would for real. Segmentation itself is done by Ani's Classic backend.
Every call is recorded in CALLS for the tests to inspect.
