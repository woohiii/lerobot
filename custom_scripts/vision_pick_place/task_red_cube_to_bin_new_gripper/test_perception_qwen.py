from custom_scripts.vision_pick_place.task_red_cube_to_bin_new_gripper.perception_qwen import (
    DEFAULT_MODEL_ID,
    model_class_name,
    model_id_from_env,
)


def test_qwen_model_id_can_use_gpu_appropriate_override(monkeypatch) -> None:
    monkeypatch.setenv("LEROBOT_QWEN_MODEL_ID", "Qwen/Qwen2-VL-2B-Instruct")
    assert model_id_from_env() == "Qwen/Qwen2-VL-2B-Instruct"
    monkeypatch.setenv("LEROBOT_QWEN_MODEL_ID", " ")
    assert model_id_from_env() == DEFAULT_MODEL_ID


def test_qwen_model_class_matches_checkpoint_family() -> None:
    assert model_class_name("Qwen/Qwen2-VL-2B-Instruct") == "Qwen2VLForConditionalGeneration"
    assert model_class_name("Qwen/Qwen2.5-VL-3B-Instruct") == "Qwen2_5_VLForConditionalGeneration"
