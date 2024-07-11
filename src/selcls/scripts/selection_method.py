


from typing import Literal


def main(
    selection_method: Literal["msp", "entropy", "odin", "doctor", "relu"],
    calibration: Literal["none", "ts", "dp"],
):
    pass


if __name__ == "__main__":
    from fire import Fire
    Fire(main)