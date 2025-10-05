"""Run full pipeline: parquetization, preprocessing, and modeling."""

from final_parquetization import parquetization
from final_preprocessing import preprocessing
from final_model import model
from final_portfolio import portfolio


def main() -> None:
    """Execute the complete data processing and modeling pipeline."""
    parquetization()
    preprocessing()
    model()
    portfolio()


if __name__ == "__main__":
    main()