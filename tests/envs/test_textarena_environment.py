from envs.textarena_env.server.environment import TextArenaEnvironment
from envs.textarena_env.models import TextArenaMessage


def test_convert_messages_coalesces_consecutive_characters():
    env = object.__new__(TextArenaEnvironment)

    raw_messages = [
        (0, "[", "PROMPT"),
        (0, "GAME", "PROMPT"),
        (0, "]", "PROMPT"),
        (1, "A", "MESSAGE"),
        (1, "B", "MESSAGE"),
        (2, "!", "MESSAGE"),
    ]

    converted = env._convert_messages(raw_messages)

    assert converted == [
        TextArenaMessage(sender_id=0, content="[GAME]", category="PROMPT"),
        TextArenaMessage(sender_id=1, content="AB", category="MESSAGE"),
        TextArenaMessage(sender_id=2, content="!", category="MESSAGE"),
    ]


def test_convert_messages_splits_on_newlines():
    env = object.__new__(TextArenaEnvironment)

    raw_messages = [
        "[",
        "G",
        "A",
        "M",
        "E",
        "]",
        "\n",
        "[",
        "N",
        "E",
        "X",
        "T",
        "]",
    ]

    converted = env._convert_messages(raw_messages)

    assert converted == [
        TextArenaMessage(sender_id=-1, content="[GAME]", category="MESSAGE"),
        TextArenaMessage(sender_id=-1, content="[NEXT]", category="MESSAGE"),
    ]


def test_convert_messages_preserves_blank_lines():
    env = object.__new__(TextArenaEnvironment)

    raw_messages = ["A", "\n", "\n", "B"]

    converted = env._convert_messages(raw_messages)

    assert converted == [
        TextArenaMessage(sender_id=-1, content="A", category="MESSAGE"),
        TextArenaMessage(sender_id=-1, content="", category="MESSAGE"),
        TextArenaMessage(sender_id=-1, content="B", category="MESSAGE"),
    ]
