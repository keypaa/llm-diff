from models.singleton import ModelManager


def test_singleton_same_instance():
    m1 = ModelManager()
    m2 = ModelManager()
    assert m1 is m2


def test_singleton_current_plan_none():
    manager = ModelManager()
    assert manager.current_plan is None


def test_singleton_clear():
    manager = ModelManager()
    manager.clear()
    assert manager.current_plan is None
