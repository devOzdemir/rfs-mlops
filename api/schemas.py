from pydantic import BaseModel, Field
from typing import Optional


class LaptopInput(BaseModel):
    # --- A. Kategorik Özellikler (Dropdown olacaklar) ---
    # Kullanıcı boş bırakabilir (Optional), biz 'most_frequent' ile dolduracağız.
    brand: Optional[str] = Field(None, example="Asus")
    platform: Optional[str] = Field(None, example="Trendyol")
    # intended_use: Optional[str] = Field(None, example="Oyun")
    # color: Optional[str] = Field(None, example="Siyah")
    # weight: Optional[str] = Field(None, example="1-1.5 kg  ")
    panel_type: Optional[str] = Field(None, example="IPS")
    # display_standard: Optional[str] = Field(None, example="FHD")

    operating_system: Optional[str] = Field(None, example="Windows 11 Home")
    cpu_family: Optional[str] = Field(None, example="Core i7")
    # cpu_model: Optional[str] = Field(None, example="13700H")
    cpu_generation: Optional[int] = Field(None, example=13)
    # cpu_cores: Optional[int] = Field(None, example=14)
    cpu_max_ghz: Optional[float] = Field(None, example=4.7)
    ram_gb: int = Field(..., example=16)
    ram_type: Optional[str] = Field(None, example="DDR5")
    ssd_gb: int = Field(..., example=512)
    # hdd_gb: Optional[int] = Field(None, example=0)

    gpu_model: Optional[str] = Field(None, example="RTX 4060")
    # gpu_type: Optional[str] = Field(None, example="Dedicated")
    gpu_vram_gb: Optional[int] = Field(None, example=8)
    # gpu_vram_type: Optional[str] = Field(None, example="GDDR6")
    screen_size_inch: float = Field(..., example=15.6)
    resolution: str = Field(..., example="1920x1080")
    # ppi: Optional[int] = Field(None, example=141)
    refresh_rate_hz: Optional[int] = Field(None, example=144)
