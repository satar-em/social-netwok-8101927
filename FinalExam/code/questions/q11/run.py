from __future__ import annotations

def run(out_dir="outputs/q11", zip_path="data.zip", data_dir="data_extracted", make_plots=True):
    from .analysis import main
    return main(out_dir=out_dir, zip_path=zip_path, data_dir=data_dir, make_plots=make_plots)
