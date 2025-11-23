from marker.scripts.convert_single import convert_single_cli
import sys 
import os 

if __name__ == "__main__":

    input_pth = sys.argv[1]
    output_pth = sys.argv[2]
    sys.argv = [
        "convert_single",
        input_pth,
        "--output_dir", output_pth,
        "--output_format", "html",
        "--use_llm",
        "--force_layout_block", "Table",
        "--converter_cls", "marker.converters.table.TableConverter",
        "--llm_service", "marker.services.gemini.GoogleGeminiService",
        "--gemini_api_key", "",
    ]

    convert_single_cli()
