from all_in_one_vision.config import LazyCall as L
from all_in_one_vision.config import instantiate


class _Dummy:
    def __init__(self, a, b=2):
        self.a = a
        self.b = b


def test_lazycall_is_declarative_not_executed():
    recipe = L(_Dummy)(a=1)
    assert "_target_" in recipe
    assert not isinstance(recipe, _Dummy)


def test_instantiate_builds_real_object():
    recipe = L(_Dummy)(a=1, b=5)
    obj = instantiate(recipe)
    assert isinstance(obj, _Dummy)
    assert obj.a == 1 and obj.b == 5


def test_instantiate_is_recursive():
    outer = L(_Dummy)(a=L(_Dummy)(a=1), b=2)
    obj = instantiate(outer)
    assert isinstance(obj, _Dummy)
    assert isinstance(obj.a, _Dummy)
    assert obj.a.a == 1
