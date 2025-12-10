"""Loader for histo work."""

from pathlib import Path

import pandas as pd

from imilia.data.paths import WSI_PATH


class IBDColEpiHistoLoader:
    def __init__(self, feats_dir: Path | None):
        self.feats_dir = feats_dir
        self.feats_paths: dict[str, Path | str] | None = None
        self.slides_paths: dict[str, Path | str] | None = None
        self.mpp_values: dict[str, float] | None = None
        self.df_metadata: pd.DataFrame | None = None

    def get_histo_feats_paths(self):
        if self.feats_paths is not None:
            return self.feats_paths
        if self.feats_dir is not None:
            print("Retrieving histology features paths...")
            feats_paths_ = list(self.feats_dir.glob("*_HE_*/features.npy"))
            feats_paths = {path.parent.name.split(".")[0]: path for path in feats_paths_}
            self.feats_paths = feats_paths
            print(f"Found {len(self.feats_paths)} paths to slide features.")
            return self.feats_paths
        else:
            raise ValueError("feats_dir must be provided to retrieve histology features paths.")

    def get_slides_paths(self):
        if self.slides_paths is not None:
            return self.slides_paths

        if self.feats_paths is None:
            self.get_histo_feats_paths()

        print("Retrieving histology slide paths...")
        slides_paths_ = {path.name.split(".ndpi")[0]: path for path in WSI_PATH.glob("*_HE_*.ndpi")}
        self.slides_paths = {key: slides_paths_[key] for key in self.feats_paths.keys()}

        print(f"Found {len(self.slides_paths)} slide paths.")

        return self.slides_paths

    def get_mpp_values(self):
        if self.mpp_values is not None:
            return self.mpp_values

        print("Retrieving MPP values from tiling tool metadata...")
        feats_paths = self.feats_paths.values()
        mpp_values = {}
        for fpath in feats_paths:
            slide_name = str(fpath.parent.name).split(".")[0]
            tt_params_path = fpath.parent / "tiling_tool_format.json"
            if tt_params_path.exists():
                tt_params = pd.read_json(tt_params_path, typ="series")
                mpp_values[slide_name] = tt_params["metadata"].get("tile_mpp", None)
        self.mpp_values = mpp_values

        return self.mpp_values

    def get_histo_metadata_info(self):
        slides_paths = self.get_slides_paths()
        feats_paths = self.get_histo_feats_paths()
        mpp_values = self.get_mpp_values()

        print("Compiling histology metadata into single dataframe...")
        df_all_slides = pd.DataFrame()
        df_all_slides["slide_path"] = slides_paths.values()
        df_all_slides["slide_name"] = df_all_slides["slide_path"].apply(lambda x: Path(x).name.split(".")[0])
        df_all_slides["feats_path"] = df_all_slides["slide_name"].apply(
            lambda x: str(next((fp for fp in feats_paths if str(fp).find(x) != -1), None))
        )
        df_all_slides["tt_tile_mpp"] = df_all_slides["slide_name"].apply(lambda x: mpp_values.get(x, None))
        df_all_slides["dataset_name"] = "IBDCOLEPI"
        self.df_metadata = df_all_slides

        return self.df_metadata


def load_data(return_as_df=False, label_col_name="inflamed", feats_dir: Path | None = None):
    hloader = IBDColEpiHistoLoader(feats_dir=feats_dir)
    feats_paths = hloader.get_histo_feats_paths()
    slides_paths = hloader.get_slides_paths()
    keys = feats_paths.keys()
    feats_paths = [feats_paths[k] for k in keys]
    slides_paths = [slides_paths[k] for k in keys]

    patient_ids = []
    labels = []
    for path in feats_paths:
        slide_name = str(path.parent.name).split(".")[0]

        slide_id = slide_name.split("_")[0]
        patient_id = slide_id
        # Label can be retrieved from the slide name, e.g.,
        # ID-97_HE_inactive.ndpi, ID-99_HE_active.ndpi
        label = slide_name.split("_")[-1]
        patient_ids.append(patient_id)
        labels.append(label)

    df_data = pd.DataFrame(
        {"slide_path": slides_paths, "patient_id": patient_ids, "features_path": feats_paths, label_col_name: labels}
    )
    df_data = df_data.set_index("patient_id")
    if not df_data.index.nunique() == len(df_data):
        print(f"Some slide names are not unique. Non-unique slides: {df_data[df_data.index.duplicated()].index.values}")

    df_data[label_col_name] = df_data[label_col_name].apply(lambda x: x == "active")
    print(f"Final dataset size: {len(df_data)} samples.")

    if return_as_df:
        return df_data
    else:
        x_paths = df_data["features_path"].to_numpy()
        y_labels = df_data[label_col_name].to_numpy()
        patient_ids = df_data.index.to_list()
        return x_paths, y_labels, patient_ids
