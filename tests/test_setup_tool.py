from src.setup_tool.main import ProgressDemo


def test_progress_demo_initial_state():
    demo = ProgressDemo(total_steps=5)

    assert demo.total_steps == 5
    assert demo.current_step == 0
    assert demo.skipped == 0
    assert demo.downloaded == 0
    assert demo.errors == 0


def test_step_result_skip_increments_skipped():
    demo = ProgressDemo()
    demo.step_result("skip", "already exists")
    assert demo.skipped == 1
    assert demo.downloaded == 0
    assert demo.errors == 0


def test_step_result_download_increments_downloaded():
    demo = ProgressDemo()
    demo.step_result("download", "downloaded successfully")
    assert demo.skipped == 0
    assert demo.downloaded == 1
    assert demo.errors == 0


def test_step_result_error_increments_errors():
    demo = ProgressDemo()
    demo.step_result("error", "something failed")
    assert demo.skipped == 0
    assert demo.downloaded == 0
    assert demo.errors == 1