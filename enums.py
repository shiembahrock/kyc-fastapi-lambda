from enum import IntEnum

class UsageStatus(IntEnum):
    Unuseable = 0
    Usable = 1
    Completed = 2

class DiscountType(IntEnum):
    Percentage = 1
    Fixed_Amount = 2

class DiscountCategory(IntEnum):
    General = 1
    Referral_Code = 2
    Buy_Credits = 3