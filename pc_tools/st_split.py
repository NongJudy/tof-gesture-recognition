"""
st_split.py
===========
ตัวแบ่งข้อมูล train/test สำหรับชุดข้อมูล ST_VL53L8CX_handposture_dataset

หัวใจของไฟล์นี้: แบ่งข้อมูลแบบ "แยกคน" (subject-independent)

ทำไมสำคัญ
---------
ST แบ่งข้อมูลแบบสับปนกันแล้วตัด 80/20 (data_loader.py:184-191)
ผลคือภาพของคนเดียวกันอยู่ทั้งฝั่ง train และ test
โมเดลจึง "เคยเห็นมือคนนี้มาแล้ว" ตอนทดสอบ -> คะแนนสูงเกินความจริง

ถ้าเอาโมเดลไปใช้กับคนใหม่ที่ไม่เคยเห็น คะแนนจะตกลง
ตัวเลขที่ใช้อ้างอิงได้จริงจึงต้องมาจากการแบ่งแบบแยกคน

โหมดการแบ่งที่มีให้
-------------------
  loso    Leave-One-Subject-Out : วนทีละคน เอา 1 คนเป็น test ที่เหลือเป็น train
          -> ได้ผลหลายรอบ รายงานค่าเฉลี่ย +/- ส่วนเบี่ยงเบน (n = จำนวนคน)
  random  แบบเดียวกับ ST : สับปนกันแล้วตัด -> ใช้เป็น baseline เพื่อเทียบเท่านั้น

ปัญหาที่ต้องจัดการ
-----------------
คลาส None มีข้อมูลจาก User1 คนเดียว
ถ้า User1 ไปอยู่ฝั่ง test โมเดลจะไม่เคยเห็นคลาส None ตอน train เลย
มี 3 ทางเลือก เลือกด้วย --none-policy :
  drop   ตัดคลาส None ออกทั้งหมด -> เหลือ 7 คลาส (แนะนำ: ตรงกับที่ ST บอกว่ารองรับ 7 ท่า)
  keep   เก็บไว้ แต่ข้ามรอบที่ User1 เป็น test (รายงานว่าข้ามไปกี่รอบ)
  force  เก็บไว้ทุกรอบ -> รอบที่ User1 เป็น test จะพังแน่นอน (ไว้ใช้สาธิตปัญหา)

วิธีรัน
-------
    python st_split.py --data-dir ./ST_VL53L8CX_handposture_dataset
    python st_split.py --data-dir ./ST_VL53L8CX_handposture_dataset --none-policy keep
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterator, Literal

import numpy as np

from st_dataset import HandPostureData, load_dataset, HAND_POSTURE_DICT

# seed มาตรฐานของโปรเจค ตั้งไว้ตรงนี้ที่เดียว ทุกการทดลองต้องใช้ค่าเดียวกัน
DEFAULT_SEED: int = 42

NonePolicy = Literal["drop", "keep", "force"]


@dataclass
class Fold:
    """ข้อมูลการแบ่ง 1 รอบ

    Attributes:
        name: ชื่อรอบ เช่น "test=User2"
        train_idx: ตำแหน่งของเฟรมที่ใช้ train
        test_idx: ตำแหน่งของเฟรมที่ใช้ test
        test_subject: รหัสคนที่เป็น test (None ถ้าแบ่งแบบสุ่ม)
    """

    name: str
    train_idx: np.ndarray
    test_idx: np.ndarray
    test_subject: str | None = None

    def __repr__(self) -> str:
        return (
            f"Fold({self.name}: train={len(self.train_idx):,} "
            f"test={len(self.test_idx):,})"
        )


@dataclass
class InnerFold:
    """การแบ่งชั้นใน 1 รอบ — ใช้เลือก hyperparameter และหยุดเทรน (EarlyStopping)

    ชั้นในนี้อยู่ข้างในรอบนอกเสมอ และ "ไม่เคยเห็น" คนที่เป็น test ของรอบนอกเลย

    Attributes:
        name: ชื่อรอบ เช่น "val=User3"
        train_idx: ตำแหน่งเฟรมที่ใช้เทรนในชั้นใน
        val_idx: ตำแหน่งเฟรมที่ใช้เป็น validation
        val_subject: รหัสคนที่เป็น validation
    """

    name: str
    train_idx: np.ndarray
    val_idx: np.ndarray
    val_subject: str

    def __repr__(self) -> str:
        return (
            f"InnerFold({self.name}: train={len(self.train_idx):,} "
            f"val={len(self.val_idx):,})"
        )


@dataclass
class NestedFold:
    """การแบ่งแบบซ้อนสองชั้น (nested) 1 รอบนอก

    ทำไมต้องซ้อนสองชั้น
    -------------------
    ST ตั้งค่าเทรนไว้ว่า `epochs: 1000` พร้อม EarlyStopping ที่เฝ้าดู `val_accuracy`
    และ ReduceLROnPlateau ที่เฝ้าดู `val_loss`
    (ที่มา: st_cnn2d_handposture_8classes_config.yaml:34-51)

    แปลว่าการเทรนใช้ชุด validation "ตัดสินใจ" 2 อย่าง:
      1. หยุดเทรนตอนไหน และย้อนกลับไปเอาน้ำหนักรอบไหน (restore_best_weights)
      2. ลด learning rate ตอนไหน

    ถ้าเอา test set ไปทำหน้าที่นี้ = เลือกโมเดลที่ทำคะแนนดีที่สุดบนข้อสอบ
    ตัวเลขที่ได้จะสูงเกินจริงและใช้อ้างอิงไม่ได้ แม้จะไม่ได้ตั้งใจโกงก็ตาม

    ลำดับการใช้งาน
    --------------
      1. วนชั้นใน (inner) เพื่อหาค่าที่ดีที่สุด เช่น จำนวน epoch ที่ควรหยุด
      2. เทรนใหม่ด้วย train_full_idx (ครบทุกคนที่ไม่ใช่ test) ด้วยค่าที่ได้จากข้อ 1
      3. วัดผลกับ test_idx เพียงครั้งเดียว ห้ามย้อนกลับไปปรับอะไรอีก

    Attributes:
        name: ชื่อรอบนอก เช่น "test=User1"
        test_subject: รหัสคนที่กันไว้เป็น test
        test_idx: ตำแหน่งเฟรมของ test — แตะได้ครั้งเดียวตอนวัดผลสุดท้าย
        train_full_idx: ทุกเฟรมที่ไม่ใช่ test ใช้เทรนรอบสุดท้าย
        inner: รายการการแบ่งชั้นใน
    """

    name: str
    test_subject: str
    test_idx: np.ndarray
    train_full_idx: np.ndarray
    inner: list[InnerFold]

    def __repr__(self) -> str:
        return (
            f"NestedFold({self.name}: train_full={len(self.train_full_idx):,} "
            f"test={len(self.test_idx):,} inner={len(self.inner)})"
        )


def subject_set(groups: np.ndarray) -> set[str]:
    """คืนเซตของรหัสคน เป็น str ธรรมดาของ Python

    ทำไมต้องมีฟังก์ชันนี้
    ---------------------
    `data.groups` เป็น numpy array การหยิบสมาชิกออกมาตรงๆ จะได้ชนิด `np.str_`
    ซึ่งใช้เปรียบเทียบและเรียงลำดับได้ปกติ แต่เวลาพิมพ์เป็นลิสต์จะแสดงว่า
    `[np.str_('User2')]` แทนที่จะเป็น `['User2']` ทำให้รายงานอ่านยาก

    `.tolist()` แปลงกลับเป็นชนิดพื้นฐานของ Python ให้ ทั้งค่าและการเปรียบเทียบ
    เหมือนเดิมทุกประการ เปลี่ยนแค่ชนิดที่ห่อไว้

    ผลพลอยได้: ทำให้ type hint ที่ประกาศว่า `str` เป็นความจริง ไม่ใช่คำโกหก

    Args:
        groups: อาร์เรย์รหัสคน

    Returns:
        เซตของรหัสคน
    """
    return set(groups.tolist())


def subject_list(groups: np.ndarray) -> list[str]:
    """คืนรายชื่อคนแบบเรียงลำดับ เป็น str ธรรมดา — ดู subject_set สำหรับเหตุผล"""
    return sorted(subject_set(groups))


def filter_classes(data: HandPostureData, keep: list[str]) -> HandPostureData:
    """คัดเฉพาะคลาสที่ต้องการ แล้วเรียงเลข index ใหม่ให้ต่อเนื่อง

    Args:
        data: ข้อมูลตั้งต้น
        keep: รายชื่อคลาสที่ต้องการเก็บ

    Returns:
        HandPostureData ที่มีเฉพาะคลาสที่เลือก

    Raises:
        ValueError: ถ้าระบุคลาสที่ไม่มีอยู่ หรือเก็บเหลือน้อยกว่า 2 คลาส
    """
    unknown = set(keep) - set(data.class_names)
    if unknown:
        raise ValueError(f"ไม่มีคลาสเหล่านี้: {sorted(unknown)}")
    if len(keep) < 2:
        raise ValueError("ต้องเหลืออย่างน้อย 2 คลาสจึงจะจำแนกได้")

    # เรียงตามเลข label ของ ST เพื่อให้ index สอดคล้องกับต้นฉบับ
    keep_sorted = sorted(keep, key=lambda c: HAND_POSTURE_DICT[c])
    old_idx = [data.class_names.index(c) for c in keep_sorted]

    mask = np.isin(data.y, old_idx)
    remap = {old: new for new, old in enumerate(old_idx)}
    y_new = np.array([remap[int(v)] for v in data.y[mask]], dtype=np.int64)

    return HandPostureData(
        X=data.X[mask],
        y=y_new,
        groups=data.groups[mask],
        class_names=keep_sorted,
    )


def find_single_subject_classes(data: HandPostureData) -> dict[str, str]:
    """หาคลาสที่มีข้อมูลจากคนเดียว

    คลาสแบบนี้ทำ subject-independent ไม่ได้ เพราะถ้าคนนั้นไปอยู่ฝั่ง test
    โมเดลจะไม่เคยเห็นคลาสนี้ตอน train

    Returns:
        dict ชื่อคลาส -> รหัสคนคนเดียวที่ทำคลาสนั้น
    """
    result: dict[str, str] = {}
    for i, cls in enumerate(data.class_names):
        users = subject_list(data.groups[data.y == i])
        if len(users) == 1:
            result[cls] = users[0]
    return result


def loso_folds(
    data: HandPostureData,
    none_policy: NonePolicy = "drop",
    verbose: bool = True,
) -> Iterator[Fold]:
    """สร้างรอบการแบ่งแบบ Leave-One-Subject-Out

    วนทีละคน: เอาคนนั้นเป็น test ที่เหลือทั้งหมดเป็น train
    ได้ผลลัพธ์เท่ากับจำนวนคน (ที่นี่คือ 4 รอบ)

    Args:
        data: ข้อมูลที่โหลดแล้ว
        none_policy: วิธีจัดการคลาสที่มีคนเดียว ("drop"/"keep"/"force")
        verbose: พิมพ์รายงานหรือไม่

    Yields:
        Fold ทีละรอบ
    """
    subjects = subject_list(data.groups)
    single = find_single_subject_classes(data)

    for subj in subjects:
        test_mask = data.groups == subj
        train_mask = ~test_mask

        # คลาสที่หายไปจากฝั่ง train เพราะเจ้าของข้อมูลไปอยู่ test
        missing = [c for c, u in single.items() if u == subj]

        if missing and none_policy == "keep":
            if verbose:
                print(
                    f"  ข้ามรอบ test={subj} : คลาส {missing} จะไม่มีใน train "
                    f"(ใช้ --none-policy drop เพื่อตัดคลาสนี้ออกแทน)"
                )
            continue

        if missing and none_policy == "force" and verbose:
            print(
                f"  เตือน: รอบ test={subj} คลาส {missing} ไม่มีในชุด train "
                f"-> โมเดลจะทายคลาสนี้ผิดทั้งหมด"
            )

        yield Fold(
            name=f"test={subj}",
            train_idx=np.flatnonzero(train_mask),
            test_idx=np.flatnonzero(test_mask),
            test_subject=subj,
        )


def missing_classes_in(
    data: HandPostureData, idx: np.ndarray
) -> list[str]:
    """หาคลาสที่ไม่มีอยู่เลยในชุดที่ระบุ

    ใช้ตรวจว่าชุด train ชุดใดชุดหนึ่งขาดคลาสไปหรือไม่
    ถ้าขาด โมเดลจะทายคลาสนั้นถูกไม่ได้เลย ไม่ว่าจะเทรนดีแค่ไหน

    Args:
        data: ข้อมูลทั้งหมด
        idx: ตำแหน่งเฟรมของชุดที่ต้องการตรวจ

    Returns:
        รายชื่อคลาสที่ไม่มีในชุดนั้น (ว่าง = ครบดี)
    """
    present = set(data.y[idx].tolist())
    return [c for i, c in enumerate(data.class_names) if i not in present]


def nested_loso_folds(
    data: HandPostureData,
    verbose: bool = True,
) -> Iterator[NestedFold]:
    """สร้างการแบ่งแบบ Nested Leave-One-Subject-Out

    โครงสร้าง เมื่อมีคน 4 คน
    ------------------------
      รอบนอก test=User1
         ชั้นใน val=User2 -> train=User3,User4
         ชั้นใน val=User3 -> train=User2,User4
         ชั้นใน val=User4 -> train=User2,User3
         เทรนสุดท้าย train_full = User2,User3,User4  -> วัดกับ User1
      (ทำแบบเดียวกันกับ User2, User3, User4 เป็น test)

    รวมการเทรน = 4 รอบนอก x 3 ชั้นใน + 4 รอบสุดท้าย = 16 ครั้ง

    หลักประกันความถูกต้อง
    ---------------------
    คนที่เป็น test ของรอบนอก ไม่ปรากฏในชั้นในเลยแม้แต่เฟรมเดียว
    ทั้ง train และ val ของชั้นในถูกตัดมาจาก train_full ซึ่งไม่มี test อยู่แล้ว

    Args:
        data: ข้อมูลที่โหลดแล้ว
        verbose: พิมพ์คำเตือนเมื่อพบคลาสขาด

    Yields:
        NestedFold ทีละรอบนอก

    Raises:
        ValueError: ถ้ามีคนน้อยกว่า 3 คน (ซ้อนสองชั้นไม่ได้)
    """
    subjects = subject_list(data.groups)
    if len(subjects) < 3:
        raise ValueError(
            f"nested LOSO ต้องมีอย่างน้อย 3 คน (มี {len(subjects)}) "
            "เพราะต้องแบ่งเป็น test / val / train อย่างละอย่างน้อย 1 คน"
        )

    for test_subj in subjects:
        test_mask = data.groups == test_subj
        train_full_idx = np.flatnonzero(~test_mask)
        remaining = [s for s in subjects if s != test_subj]

        inner: list[InnerFold] = []
        for val_subj in remaining:
            # ตัดจาก train_full เท่านั้น จึงไม่มีทางปน test เข้ามา
            val_mask = data.groups == val_subj
            inner_train_idx = np.flatnonzero(~test_mask & ~val_mask)
            inner_val_idx = np.flatnonzero(val_mask)

            if verbose:
                gone = missing_classes_in(data, inner_train_idx)
                if gone:
                    print(
                        f"  เตือน: test={test_subj} val={val_subj} "
                        f"-> ชุด train ชั้นในขาดคลาส {gone}"
                    )

            inner.append(
                InnerFold(
                    name=f"val={val_subj}",
                    train_idx=inner_train_idx,
                    val_idx=inner_val_idx,
                    val_subject=val_subj,
                )
            )

        if verbose:
            gone = missing_classes_in(data, train_full_idx)
            if gone:
                print(f"  เตือน: test={test_subj} -> ชุดเทรนสุดท้ายขาดคลาส {gone}")

        yield NestedFold(
            name=f"test={test_subj}",
            test_subject=test_subj,
            test_idx=np.flatnonzero(test_mask),
            train_full_idx=train_full_idx,
            inner=inner,
        )


def random_val_from_train(
    train_idx: np.ndarray,
    data: HandPostureData,
    val_size: float = 0.2,
    seed: int = DEFAULT_SEED,
    stratify: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """แบ่ง validation ออกจาก train แบบสุ่ม — ใช้ทำ ablation เท่านั้น

    นี่คือ "ทาง A" ที่ง่ายกว่า nested แต่มีข้อบกพร่อง:
    validation จะมีคนคนเดียวกับ train ทำให้คะแนน validation สูงเกินจริง
    EarlyStopping จึงตัดสินใจบนตัวเลขที่หลอกตา

    เก็บไว้เพื่อรายงานในวิทยานิพนธ์ว่า "ถ้าใช้วิธีง่ายกว่านี้ ผลต่างเท่าไหร่"
    ห้ามใช้เป็นวิธีหลัก

    Args:
        train_idx: ตำแหน่งเฟรมฝั่ง train ของรอบนอก
        data: ข้อมูลทั้งหมด (ใช้ดู label ตอน stratify)
        val_size: สัดส่วนที่กันเป็น validation
        seed: ค่าสุ่มคงที่
        stratify: คุมสัดส่วนคลาสให้เท่ากันทั้งสองฝั่งหรือไม่

    Returns:
        (train_idx ใหม่, val_idx)

    Raises:
        ValueError: ถ้า val_size ไม่อยู่ในช่วง (0, 1)
    """
    if not 0.0 < val_size < 1.0:
        raise ValueError(f"val_size ต้องอยู่ระหว่าง 0 ถึง 1 (ได้ {val_size})")

    rng = np.random.default_rng(seed)

    if not stratify:
        shuffled = rng.permutation(train_idx)
        n_val = int(round(val_size * len(shuffled)))
        return shuffled[n_val:], shuffled[:n_val]

    # แบ่งทีละคลาส เพื่อให้สัดส่วนคลาสสองฝั่งใกล้เคียงกัน
    tr_parts: list[np.ndarray] = []
    va_parts: list[np.ndarray] = []
    for cls_i in range(len(data.class_names)):
        cls_idx = train_idx[data.y[train_idx] == cls_i]
        if len(cls_idx) == 0:
            continue
        shuffled = rng.permutation(cls_idx)
        n_val = int(round(val_size * len(shuffled)))
        va_parts.append(shuffled[:n_val])
        tr_parts.append(shuffled[n_val:])

    return np.concatenate(tr_parts), np.concatenate(va_parts)


def check_nested_no_leakage(fold: NestedFold, data: HandPostureData) -> bool:
    """ตรวจการรั่วไหลของการแบ่งแบบซ้อนสองชั้น — ตรวจ 4 ข้อ

    ข้อ 1  test ไม่อยู่ใน train_full
    ข้อ 2  test ไม่อยู่ในชั้นในเลย ไม่ว่าฝั่ง train หรือ val
    ข้อ 3  ในแต่ละชั้นใน val ไม่ทับกับ train
    ข้อ 4  ชั้นใน train + val รวมกันแล้วเท่ากับ train_full พอดี (ไม่มีเฟรมหาย)

    ข้อ 2 คือข้อที่สำคัญที่สุด ถ้าพลาดข้อนี้ EarlyStopping จะเห็นข้อมูล test
    ทางอ้อม และตัวเลขสุดท้ายจะใช้ไม่ได้ทั้งหมด

    Returns:
        True ถ้าผ่านทั้ง 4 ข้อ
    """
    ok = True
    test_users = subject_set(data.groups[fold.test_idx])

    # ข้อ 1
    if test_users & subject_set(data.groups[fold.train_full_idx]):
        print(f"  ไม่ผ่าน ข้อ 1: คน test อยู่ใน train_full ด้วย")
        ok = False

    for inner in fold.inner:
        inner_users = subject_set(data.groups[inner.train_idx]) | subject_set(
            data.groups[inner.val_idx]
        )

        # ข้อ 2
        if test_users & inner_users:
            print(f"  ไม่ผ่าน ข้อ 2: คน test โผล่ในชั้นใน {inner.name}")
            ok = False

        # ข้อ 3
        if subject_set(data.groups[inner.train_idx]) & subject_set(data.groups[inner.val_idx]):
            print(f"  ไม่ผ่าน ข้อ 3: train กับ val ทับกันใน {inner.name}")
            ok = False

        # ข้อ 4 — เทียบเป็นเซตของตำแหน่งเฟรม ไม่ใช่แค่จำนวน
        combined = set(inner.train_idx.tolist()) | set(inner.val_idx.tolist())
        if combined != set(fold.train_full_idx.tolist()):
            print(f"  ไม่ผ่าน ข้อ 4: ชั้นใน {inner.name} รวมกันแล้วไม่เท่ากับ train_full")
            ok = False

    return ok


def describe_nested(fold: NestedFold, data: HandPostureData) -> None:
    """พิมพ์รายละเอียดของรอบนอก 1 รอบ พร้อมชั้นในทั้งหมด"""
    print(f"\n{fold.name}")
    print(
        f"  เทรนสุดท้าย {len(fold.train_full_idx):>6,} เฟรม | "
        f"test {len(fold.test_idx):>6,} เฟรม"
    )
    for inner in fold.inner:
        print(
            f"    {inner.name:<12} train {len(inner.train_idx):>6,} | "
            f"val {len(inner.val_idx):>6,}"
        )


def random_fold(
    data: HandPostureData,
    test_size: float = 0.2,
    seed: int = DEFAULT_SEED,
) -> Fold:
    """แบ่งแบบสุ่มปนกัน — เลียนแบบวิธีของ ST เพื่อใช้เป็น baseline เทียบ

    ห้ามใช้ตัวเลขจากการแบ่งแบบนี้เป็นผลหลักในวิทยานิพนธ์
    มีไว้เพื่อแสดงว่า "ถ้าแบ่งแบบนี้จะได้คะแนนสูงเกินจริงเท่าไหร่"

    ต่างจาก ST ตรงที่กำหนด seed ไว้ ทำให้ทำซ้ำได้ (ของ ST ไม่ได้กำหนด)

    Args:
        data: ข้อมูลที่โหลดแล้ว
        test_size: สัดส่วนที่ใช้เป็น test
        seed: ค่าสุ่มคงที่ เพื่อให้ผลเหมือนเดิมทุกครั้ง

    Returns:
        Fold เดียว
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(data))
    n_test = int(round(test_size * len(data)))
    return Fold(
        name=f"random(seed={seed})",
        train_idx=idx[n_test:],
        test_idx=idx[:n_test],
        test_subject=None,
    )


def describe_fold(fold: Fold, data: HandPostureData) -> None:
    """พิมพ์รายละเอียดของ 1 รอบ: ขนาด และจำนวนต่อคลาสทั้งสองฝั่ง"""
    print(f"\n{fold.name}")
    print(f"  train {len(fold.train_idx):>6,} เฟรม | test {len(fold.test_idx):>6,} เฟรม")

    y_tr, y_te = data.y[fold.train_idx], data.y[fold.test_idx]
    print(f"  {'คลาส':<14}{'train':>8}{'test':>8}")
    for i, cls in enumerate(data.class_names):
        n_tr, n_te = int((y_tr == i).sum()), int((y_te == i).sum())
        flag = "  <-- ไม่มีใน train!" if n_tr == 0 else ""
        print(f"  {cls:<14}{n_tr:>8,}{n_te:>8,}{flag}")


def check_no_leakage(fold: Fold, data: HandPostureData) -> bool:
    """ตรวจว่าไม่มีคนคนเดียวกันอยู่ทั้ง train และ test

    นี่คือการตรวจที่สำคัญที่สุดของทั้งไฟล์
    ถ้าตรวจไม่ผ่าน แปลว่าการแบ่งข้อมูลมีข้อบกพร่อง ผลที่ได้เชื่อไม่ได้

    Returns:
        True ถ้าไม่มีการรั่วไหล
    """
    train_users = subject_set(data.groups[fold.train_idx])
    test_users = subject_set(data.groups[fold.test_idx])
    overlap = train_users & test_users
    if overlap:
        print(f"  ตรวจไม่ผ่าน: คนเหล่านี้อยู่ทั้งสองฝั่ง -> {sorted(overlap)}")
        return False
    return True


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="แบ่งข้อมูลแบบแยกคน (subject-independent) และเทียบกับวิธีสุ่มของ ST"
    )
    parser.add_argument("--data-dir", required=True, help="path ของโฟลเดอร์ชุดข้อมูล")
    parser.add_argument(
        "--none-policy",
        choices=["drop", "keep", "force"],
        default="drop",
        help="จัดการคลาสที่มีข้อมูลจากคนเดียวอย่างไร (ค่าเริ่มต้น: drop)",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="สัดส่วน validation สำหรับ ablation ทาง A (ค่าเริ่มต้น 0.2)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    print("=" * 62)
    print("โหลดข้อมูล")
    print("=" * 62)
    data = load_dataset(args.data_dir)

    single = find_single_subject_classes(data)
    if single:
        print(f"\nคลาสที่มีข้อมูลจากคนเดียว: {single}")

    if args.none_policy == "drop" and single:
        keep = [c for c in data.class_names if c not in single]
        print(f"นโยบาย drop -> ตัดออก {list(single)} เหลือ {len(keep)} คลาส")
        data = filter_classes(data, keep)
        print(f"เหลือ {len(data):,} เฟรม")

    print("\n" + "=" * 62)
    print("การแบ่งแบบแยกคน (Leave-One-Subject-Out) -- ผลหลักที่ใช้อ้างอิงได้")
    print("=" * 62)

    folds = list(loso_folds(data, none_policy=args.none_policy))
    all_ok = True
    for fold in folds:
        describe_fold(fold, data)
        ok = check_no_leakage(fold, data)
        all_ok &= ok
        print(f"  ตรวจการรั่วไหลของข้อมูล: {'ผ่าน' if ok else 'ไม่ผ่าน'}")

    print(f"\nจำนวนรอบทั้งหมด: {len(folds)}  (n = {len(folds)} สำหรับรายงานค่าเฉลี่ย)")
    print(f"ผลการตรวจรวม: {'ผ่านทุกรอบ' if all_ok else 'มีรอบที่ไม่ผ่าน'}")

    if len(folds) < 3:
        print(
            "  เตือน: n < 3 ยังสรุปทางสถิติไม่ได้ "
            "ต้องเก็บข้อมูลจากคนเพิ่มหรือรายงานเป็นรายรอบ"
        )

    print("\n" + "=" * 62)
    print("การแบ่งแบบซ้อนสองชั้น (Nested LOSO) -- วิธีหลักของโปรเจคนี้")
    print("=" * 62)
    print(
        "ชั้นนอก: กัน 1 คนเป็น test\n"
        "ชั้นใน : วนกันอีก 1 คนเป็น validation สำหรับ EarlyStopping\n"
        "จากนั้นเทรนใหม่ด้วยทุกคนที่ไม่ใช่ test แล้ววัดผลครั้งเดียว"
    )

    nested = list(nested_loso_folds(data))
    n_train_runs = sum(len(f.inner) for f in nested) + len(nested)
    nested_ok = True
    for fold in nested:
        describe_nested(fold, data)
        ok = check_nested_no_leakage(fold, data)
        nested_ok &= ok
        print(f"  ตรวจการรั่วไหล 4 ข้อ: {'ผ่าน' if ok else 'ไม่ผ่าน'}")

    print(f"\nรอบนอกทั้งหมด: {len(nested)}  (n = {len(nested)} สำหรับรายงานค่าเฉลี่ย)")
    print(f"จำนวนการเทรนที่ต้องรันทั้งหมด: {n_train_runs} ครั้ง")
    print(f"ผลการตรวจรวม: {'ผ่านทุกรอบ' if nested_ok else 'มีรอบที่ไม่ผ่าน'}")

    if len(nested) < 3:
        print("  เตือน: n < 3 ยังสรุปทางสถิติไม่ได้ ต้องรายงานเป็นรายรอบ")

    print("\n" + "=" * 62)
    print("Ablation: แบ่ง validation แบบสุ่มจาก train (ทาง A) -- ไว้เทียบเท่านั้น")
    print("=" * 62)
    for fold in nested:
        tr, va = random_val_from_train(
            fold.train_full_idx, data, val_size=args.val_size, seed=args.seed
        )
        va_users = subject_list(data.groups[va])
        tr_users = subject_list(data.groups[tr])
        shared = set(tr_users) & set(va_users)
        print(
            f"  {fold.name:<14} train {len(tr):>6,} | val {len(va):>6,} | "
            f"คนที่อยู่ทั้ง train และ val: {sorted(shared) if shared else 'ไม่มี'}"
        )
    print(
        "\n  หมายเหตุ: การที่คนซ้ำกันสองฝั่งคือข้อบกพร่องของวิธีนี้โดยตั้งใจ\n"
        "  คะแนน validation จะสูงเกินจริง ทำให้ EarlyStopping หยุดผิดจังหวะ\n"
        "  รายงานเป็น ablation เพื่อแสดงว่าทำไมจึงเลือก nested"
    )

    print("\n" + "=" * 62)
    print("การแบ่งแบบสุ่ม (เลียนแบบ ST) -- baseline สำหรับเทียบเท่านั้น")
    print("=" * 62)
    rf = random_fold(data, test_size=args.test_size, seed=args.seed)
    describe_fold(rf, data)
    ok = check_no_leakage(rf, data)
    print(f"  ตรวจการรั่วไหลของข้อมูล: {'ผ่าน' if ok else 'ไม่ผ่าน (คาดไว้แล้ว)'}")
    print(
        "\n  หมายเหตุ: การแบ่งแบบนี้ 'ไม่ผ่าน' เป็นเรื่องปกติและเป็นสิ่งที่ต้องการแสดง\n"
        "  เพราะจุดประสงค์คือชี้ให้เห็นว่าวิธีของ ST ทำให้คนเดียวกันอยู่ทั้งสองฝั่ง"
    )


if __name__ == "__main__":
    _main()

# ─────────────────────────────────────────────────────────────────────────
# ข้อจำกัดที่ทราบ (failure modes)
#
# 1. มีคนแค่ 4 คน -> LOSO ได้ n = 4 และชั้นในได้แค่ 3
#    n น้อยแบบนี้ค่าเบี่ยงเบนจะกว้าง ต้องรายงานทุกรอบ ไม่ใช่เฉพาะค่าเฉลี่ย
#    ถ้าเก็บข้อมูลเองใน Phase 5 ควรมีอย่างน้อย 6 คน
#
# 2. User1 มีข้อมูล 43% ของทั้งหมด รอบที่ User1 เป็น test จึงมี train น้อยกว่ารอบอื่นมาก
#    ทำให้เทียบระหว่างรอบไม่แฟร์นัก ต้องระบุไว้ในวิทยานิพนธ์
#
# 3. [แก้แล้ว 4 Sep] validation set — ใช้ nested LOSO เป็นวิธีหลัก
#    และเก็บวิธีสุ่มจาก train ไว้เป็น ablation
#    เหตุผลที่ต้องมี: ST ตั้ง EarlyStopping(monitor=val_accuracy) และ
#    ReduceLROnPlateau(monitor=val_loss) ไว้ในไฟล์ตั้งค่า
#    (st_cnn2d_handposture_8classes_config.yaml:34-51)
#    ถ้าไม่มี validation แยก ต้องเอา test ไปทำหน้าที่นี้ = ผลใช้ไม่ได้
#
# 4. ยังไม่มีการทำ class weight ทั้งที่ข้อมูลไม่สมดุล (Fist 19.9% vs None 3.3%)
#    ต้องตัดสินใจตอนเขียนสคริปต์เทรน และต้องคำนวณจากฝั่ง train เท่านั้น
#    ห้ามคำนวณจากข้อมูลทั้งหมด เพราะจะเป็นการดูข้อมูล test ทางอ้อม
#
# 5. random_fold ไม่ได้ทำ stratify โดยตั้งใจ เพราะต้องการเลียนแบบวิธีของ ST
#    ซึ่งใช้ shuffle แล้ว take/skip เฉยๆ (data_loader.py:184-191)
#    ผลคือสัดส่วนคลาสสองฝั่งไม่เท่ากัน — นั่นคือลักษณะของวิธีเขาจริงๆ
#    ส่วน random_val_from_train ทำ stratify เป็นค่าเริ่มต้น เพราะไม่ได้เลียนแบบใคร
#
# 6. ยังไม่ได้เลียนแบบ shuffle ของ TensorFlow แบบบิตต่อบิต
#    random_fold ใช้ numpy ผลลัพธ์จึงไม่ตรงกับ ST เป๊ะ
#    ในวิทยานิพนธ์ต้องเขียนว่า "เลียนแบบเชิงวิธี" ไม่ใช่ "ผลเหมือนกันทุกเฟรม"
# ─────────────────────────────────────────────────────────────────────────
