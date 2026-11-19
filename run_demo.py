import os
from dotenv import load_dotenv
from src.devin_qa import run_devin_qa, save_qa_report
from src.pipeline import (
    build_mask_image,
    create_sample_manga_panel,
    render_translated_panel,
    translate_regions,
)


def main() -> None:
    load_dotenv()
    os.makedirs("output", exist_ok=True)

    jp_path = os.path.join("output", "sample_jp.png")
    mask_path = os.path.join("output", "sample_masked.png")
    en_path = os.path.join("output", "sample_en.png")
    qa_path = os.path.join("output", "qa_report.json")

    base_img, regions = create_sample_manga_panel(jp_path)
    build_mask_image(base_img, regions, mask_path)
    regions = translate_regions(regions)
    render_translated_panel(base_img, regions, en_path)

    qa_report = run_devin_qa(regions)
    save_qa_report(qa_report, qa_path)

    print("Pipeline completed successfully.")
    print(f"- Original panel : {jp_path}")
    print(f"- Mask preview   : {mask_path}")
    print(f"- Translated out : {en_path}")
    print(f"- Devin QA report: {qa_path} (status={qa_report['status']})")


if __name__ == "__main__":
    main()
