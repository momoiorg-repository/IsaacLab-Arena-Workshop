# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The per-episode instruction, and the material identity it names.

**Why this file exists.** Until now every chuck demo carried the SAME sentence -- "Pick up a
workpiece from the tray and lift it clear." -- while the environment picked a different one of the
three parts each episode. Identical instruction, different target: the task was not well posed, and
a policy trained on it could only guess. The instruction has to say WHICH part.

**Why the colour and not the size.** Selection and adaptation are split. The instruction selects
by COLOUR, which is what a camera can actually resolve -- the external views run at 3.4 mm per
pixel, where W-B's Ø25->Ø32 step is 2.1 pixels and is not a usable cue. The DIMENSIONS then drive
what happens downstream (grip station, insertion depth, seating tolerance), which is where §0-3's
claim about continuous dimensions belongs: the loader adapts to a size it was never told.

That keeps the appearance randomisation honest in both directions. Appearance is still drawn
INDEPENDENTLY of the variant, so it is not a family label and cannot be used to shortcut the
dimension handling; and because it is independent, naming it selects a part without leaking which
variant that part is.

**Vocabulary.** One phrase per MDL in `materials.yaml`. The mapping is explicit rather than derived
from the file name: `Wood_Tiles_Oak_Mountain` is "oak" to a person and "wood tiles oak mountain" to
a string-splitter, and the sentence a policy is trained on should read like an instruction.
"""

from __future__ import annotations

#: MDL stem -> the noun phrase used in the instruction. Held-out materials are here too: they are
#: never drawn during training, but an evaluation run has to be able to name them.
MATERIAL_PHRASE = {
    # Named by COLOUR, not by material. "Carpaint_Metallic_01" is a file name; "blue" is what the
    # camera sees and what the sentence has to mean. The library was chosen for exactly this -- four
    # colours that separate at the 3.4 mm/px the external views run at, replacing six woods and six
    # metals whose names were unique and whose renders were not.
    "Carpaint_Metallic_01": "blue",
    "Carpaint_Metallic_02": "red",
    "Carpaint_Metallic_04": "silver",
    "Carpaint_Metallic_06": "gold",
    # held out -- never drawn during training, named so an evaluation run can ask for them
    "Carpaint_Metallic_03": "green",
    "Carpaint_Metallic_05": "black",
    "Carpaint_Metallic_07": "white",
    "Carpaint_Metallic_08": "orange",
}

#: Stable index per material, so the identity can ride in a fixed-width observation channel.
MATERIAL_INDEX = {name: i for i, name in enumerate(sorted(MATERIAL_PHRASE))}

#: Family index, coarser than the material and the level a policy is most likely to generalise at.
FAMILY_INDEX = {"carpaint": 0, "held_out": 1}

#: One template per slice, keyed by what the demo actually ends with. The sentence is the label:
#: a demo that loads the chuck labelled "stand it on the pad" teaches the words to mean the wrong
#: motion -- which is exactly what the first `--until loaded` smoke run recorded.
TEMPLATES = {
    "place": "Pick up the {phrase} workpiece and stand it on the pad.",
    "load": "Pick up the {phrase} workpiece and load it into the chuck.",
    # Hierarchy-upstream slice: the demo ends at the lift, and the sentence must describe exactly
    # that (v5 lesson: a verb the demo does not perform poisons the conditioning).
    "lift": "Pick up the {phrase} workpiece and lift it clear.",
}
TEMPLATE = TEMPLATES["place"]  # backward-compatible name


def phrase_for(entry: dict | None) -> str:
    """The noun phrase for one drawn appearance, or a generic one when nothing was drawn.

    The fallback is deliberately still a valid instruction rather than an error: a run with
    appearance randomisation switched off (`BDASH_PLAIN_MATERIALS`) is a legitimate control, and it
    should produce demos that are merely unselective, not demos that are broken.
    """
    if not entry:
        return ""
    name = entry.get("name") or ""
    return MATERIAL_PHRASE.get(name, name.replace("_", " ").lower())


def instruction_for(entry: dict | None, action: str = "place") -> str:
    template = TEMPLATES[action]
    phrase = phrase_for(entry)
    if phrase:
        return template.format(phrase=phrase)
    return template.format(phrase="").replace("the  workpiece", "a workpiece")


def identity_of(entry: dict | None) -> tuple[int, int]:
    """``(material_index, family_index)`` for the observation channel; ``(-1, -1)`` if unset.

    An INDEX, not the RGB the old channel carried. An MDL has no scalar colour to embed -- the old
    code wrote five zeros for exactly this reason and said so in a comment -- and the thing worth
    recording is which material it was, which an index says exactly and a colour only approximates.
    """
    if not entry:
        return -1, -1
    return (
        MATERIAL_INDEX.get(entry.get("name") or "", -1),
        FAMILY_INDEX.get(entry.get("family") or "", -1),
    )
