"""Miner-side checks that track the 2026-11-11 validator update (#125, #126).

Run with miner_env: ``python -m pytest unittest/test_miner_word_check_and_commitments.py``
"""
import hashlib

import pytest
from bittensor_wallet import Keypair

from MIID.miner import word_check as wc


def _check(target_text, heard, language):
    words = target_text.split()
    return wc.align(
        wc.canonical_targets(words, language),
        wc.canonical_heard(heard, language, words),
    )


# Transcripts as Whisper wrote them for correctly spoken clips (miner log, 2026-11-11).
@pytest.mark.parametrize("target, heard, language", [
    ("5 H P O shuttle author seven roof five empty circuit rise badge pear classic cookie R D 8 L",
     "5 HP O'Shuttle Author 7 Roof 5 Empty Circuit Rise Badge Pair Classic Cookie RD8L", "en"),
    ("5 H P O skylight hope choir seven scarf wool nerve boot screw cork one latch R D 8 L",
     "5 HP O Skylight Hope Quire 7 Scarf Wool Nerve Boot Screw Cork 1 Latch RD8L", "en"),
    ("5 H P O note one ingredient bruise cradle carrot dessert two concrete judge sand box R D 8 L",
     "5. H.P.O. Note. 1. Ingredient. Brews. Cradle. Carrot. Dessert. 2. Concrete. Judge. Sand. Box. R.D. 8. L.", "en"),
    ("5 H P O yeso caliente lino porción siete pelota cama posada rol sidra tres metálico R D 8 L",
     "5 HP o yeso caliente lino porción 7 pelota cama posada roll sidra 3 metálico R de 8L.", "es"),
    ("5 H P O cucharón cima nueve puente cuatro melón clavo tornillo claraboya silbato estación lleno R D 8 L",
     "5 HP o Cucharón, Cima, 9 Puente, 4 Melón, Clavo, Tornillo, Claraboya, Silvato, Estación Lleno, RD8L", "es"),
    ("5 H P O ola forma mano tres cuaderno chispa kayak siete campus barril marcador corcho R D 8 L",
     "5 HP o hola, forma, mano, 3, cuaderno, chispa, kayak, 7, campus, barril, marcador, corcho, RD, 8, L.", "es"),
    # An English digit word written as its homophone.
    ("5 H P O four apple R D 8 L", "5 HPO for apple RD8L", "en"),
])
def test_homophones_count_as_the_target_word(target, heard, language):
    assert _check(target, heard, language)["ok"]


@pytest.mark.parametrize("target, heard, language", [
    ("5 H P O skylight hope choir R D 8 L", "5 HPO Skylight Hoke Quire RD8L", "en"),
    ("5 H P O cloud book mountain paper R D 8 L", "5HPO Cloud Book Mount Paper RD8L", "en"),
    ("5 H P O cucharón tornillo R D 8 L", "5 HP o Cucharón, Tormillo, RD8L", "es"),
    ("5 H P O fiesta lima historia R D 8 L", "5 HP o fiesta, anima, historia, RD8L", "es"),
    # Seseo is not merged: Spain keeps casa and caza apart.
    ("5 H P O caza R D 8 L", "5 HP o casa RD8L", "es"),
    # Nor r and rr.
    ("5 H P O perro R D 8 L", "5 HP o pero RD8L", "es"),
])
def test_real_word_errors_still_fail(target, heard, language):
    assert not _check(target, heard, language)["ok"]


def test_hotkey_letter_wins_over_a_pool_homophone():
    # "see" names the hotkey letter C; it must not be taken for the pool word "sea".
    result = _check("C H P O sea apple R D 8 L", "see HPO sea apple RD8L", "en")
    assert result["ok"], result


@pytest.mark.parametrize("a, b", [
    ("vaca", "baca"), ("hola", "ola"), ("llama", "yama"), ("gente", "jente"),
    ("queso", "keso"), ("rey", "rei"), ("cielo", "zielo"), ("rol", "roll"),
])
def test_spanish_sound_key_merges(a, b):
    assert wc._spanish_sound(a) == wc._spanish_sound(b)


@pytest.mark.parametrize("a, b", [
    ("casa", "caza"), ("pero", "perro"), ("cien", "sien"), ("chico", "cico"), ("guerra", "gerra"),
])
def test_spanish_sound_key_keeps_apart(a, b):
    assert wc._spanish_sound(a) != wc._spanish_sound(b)


def test_homophone_table_covers_the_validator_pool():
    from MIID.validator.voice_words import ENGLISH_WORDS

    table = wc._homophones_en()
    assert table, "MIID/miner/homophones_en.json is missing or empty"
    covered = [w for w in ENGLISH_WORDS if w in table]
    assert len(covered) > 300
    assert table["pear"] == ["pair", "pare"]


def test_miner_commitments_pass_the_validator_check(tmp_path, monkeypatch):
    from MIID.miner import s3_upload
    from MIID.miner.submission_builder import (
        build_path_signature, commitment_error, sign_media_hash,
    )
    from MIID.protocol import S3Submission

    monkeypatch.setattr(s3_upload, "USE_S3", False)
    monkeypatch.setattr(s3_upload, "LOCAL_STORAGE_DIR", tmp_path)
    key = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
    challenge = "challenge_1794341017_5GQqAhLK"
    payload = b"synthetic image bytes"
    digest = hashlib.sha256(payload).hexdigest()
    path_signature = build_path_signature(key, challenge)
    signature = sign_media_hash(key, challenge, digest)

    for variation_type, seed, ext in (
        ("lighting_edit+pose_edit", "0e67e82821d7_f", ".png"),
        ("voice", "voice_M3_095", ".wav"),
    ):
        s3_key = s3_upload.upload_to_s3(
            encrypted_data=payload, miner_hotkey=key.ss58_address, signature=signature,
            image_hash=digest, target_round=1, challenge_id=challenge,
            variation_type=variation_type, path_signature=path_signature,
            seed_image_name=seed, source_ext=ext,
        )
        submission = S3Submission(
            s3_key=s3_key, image_hash=digest, signature=signature,
            variation_type=variation_type, path_signature=path_signature,
        )
        assert commitment_error(challenge, key.ss58_address, submission, payload) is None

    # And it reports what the validator would reject.
    other = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
    assert commitment_error(challenge, other.ss58_address, submission, payload)
    assert commitment_error(challenge, key.ss58_address, submission, b"different bytes")
