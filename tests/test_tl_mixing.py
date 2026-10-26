import os
import tempfile
import unittest

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

import SCTools as sct

GROUPS = ["Astro", "EN", "IN", "Oligo", "OPC"]
LINEAGE = {"Astro": "Astro", "EN": "Neuron", "IN": "Neuron", "Oligo": "Oligo", "OPC": "OPC"}


def _profiles(rng, n_shared=100, n_markers=20):
    n_genes = n_shared + n_markers * len(GROUPS)
    base = rng.gamma(2.0, 1.0, size=n_genes)
    base[n_shared:] *= 0.05
    profiles = {}
    for i, group in enumerate(GROUPS):
        p = base.copy()
        p[n_shared + i * n_markers : n_shared + (i + 1) * n_markers] *= 400
        profiles[group] = p / p.sum()
    return profiles, n_genes


def _make_adata(n_pure=150, n_doublet=25, n_ambient=25, n_opc_oligo=25, seed=0):
    rng = np.random.default_rng(seed)
    profiles, n_genes = _profiles(rng)
    pooled = np.mean([profiles[g] for g in GROUPS], axis=0)
    rows, labels, kinds = [], [], []

    def draw(p, depth):
        return rng.multinomial(depth, p)

    for group in GROUPS:
        for _ in range(n_pure):
            rows.append(draw(profiles[group], 2000))
            labels.append(group)
            kinds.append("pure")
    for _ in range(n_doublet):
        rows.append(draw(profiles["Astro"], 2000) + draw(profiles["EN"], 2000))
        labels.append("Astro")
        kinds.append("doublet")
    for _ in range(n_ambient):
        rows.append(draw(0.4 * profiles["Astro"] + 0.6 * pooled, 2500))
        labels.append("Astro")
        kinds.append("ambient")
    for _ in range(n_opc_oligo):
        rows.append(draw(0.5 * profiles["OPC"] + 0.5 * profiles["Oligo"], 2000))
        labels.append("OPC")
        kinds.append("opc_oligo")

    counts = sparse.csr_matrix(np.vstack(rows).astype(np.float32))
    obs = pd.DataFrame(
        {
            "subclass": pd.Categorical(labels),
            "class": pd.Categorical([LINEAGE[x] for x in labels]),
            "kind": kinds,
            "channel": rng.choice(["lib1", "lib2"], size=len(labels)),
        },
        index=[f"cell{i}" for i in range(len(labels))],
    )
    var = pd.DataFrame(index=[f"gene{i}" for i in range(n_genes)])
    return ad.AnnData(counts, obs=obs, var=var)


class DetectLineageMixingTests(unittest.TestCase):
    def test_doublets_are_flagged_with_top_foreign_group(self):
        adata = _make_adata()

        summary = sct.tl.detect_lineage_mixing(
            adata, groupby="subclass", lineage_key="class", use_raw=False, n_markers=20
        )

        kind = adata.obs["kind"]
        flag = adata.obs["mixing_flag"].astype(bool)
        self.assertGreaterEqual(flag[kind == "doublet"].mean(), 0.95)
        self.assertLessEqual(flag[kind == "pure"].mean(), 0.02)
        foreign_types = adata.obs.loc[kind == "doublet", "mixing_top"]
        self.assertGreaterEqual((foreign_types == "EN").mean(), 0.9)
        self.assertEqual(adata.obsm["mixing_weights"].shape, (adata.n_obs, len(GROUPS)))
        self.assertIn("group_summary", adata.uns["mixing"])
        self.assertListEqual(
            list(summary.columns), ["group", "n_cells", "median_UMI_frac", "n_flagged", "flag_fraction", "top"]
        )
        self.assertEqual(summary.set_index("group").loc["Astro", "top"], "EN")

    def test_doublet_vs_ambient_source(self):
        adata = _make_adata()

        sct.tl.detect_lineage_mixing(
            adata, groupby="subclass", lineage_key="class", use_raw=False, n_markers=20, library_key="channel"
        )

        obs = adata.obs
        doublet_source = obs.loc[(obs["kind"] == "doublet") & (obs["mixing_flag"] == 1), "mixing_source"]
        ambient_source = obs.loc[(obs["kind"] == "ambient") & (obs["mixing_flag"] == 1), "mixing_source"]
        self.assertGreater(len(doublet_source), 0)
        self.assertGreater(len(ambient_source), 0)
        self.assertGreaterEqual((doublet_source == "doublet").mean(), 0.9)
        self.assertGreaterEqual((ambient_source == "ambient").mean(), 0.9)
        self.assertTrue(obs.loc[obs["mixing_flag"] == 0, "mixing_doublet_logLR"].isna().all())
        self.assertIn("library_summary", adata.uns["mixing"])

    def test_known_lineage_sets_protects_transition_states(self):
        adata = _make_adata()
        transition = adata.obs["kind"] == "opc_oligo"

        sct.tl.detect_lineage_mixing(adata, lineage_key="class", use_raw=False, n_markers=20)
        self.assertGreaterEqual(adata.obs.loc[transition, "mixing_flag"].mean(), 0.9)

        sct.tl.detect_lineage_mixing(
            adata, lineage_key="class", known_lineage_sets=[("OPC", "Oligo")], use_raw=False, n_markers=20
        )
        self.assertEqual(int(adata.obs.loc[transition, "mixing_flag"].sum()), 0)

    def test_streamed_matrix_matches_in_memory(self):
        adata = _make_adata(n_pure=60, n_doublet=10, n_ambient=0, n_opc_oligo=0)
        adata.layers["counts_csc"] = sparse.csc_matrix(adata.X)

        # chunk_size also sets EM blocks, so converge tightly to compare the two paths
        converge = dict(tol=1e-9, max_iter=5000)
        sct.tl.detect_lineage_mixing(adata, lineage_key="class", use_raw=False, n_markers=20, key_added="mem", **converge)
        sct.tl.detect_lineage_mixing(
            adata, lineage_key="class", layer="counts_csc", n_markers=20, chunk_size=37, key_added="stream", **converge
        )

        np.testing.assert_allclose(adata.obs["mem_UMI_frac"], adata.obs["stream_UMI_frac"], atol=1e-3)

    def test_small_groups_are_left_unscored(self):
        adata = _make_adata(n_pure=60, n_doublet=0, n_ambient=0, n_opc_oligo=0)
        labels = adata.obs["subclass"].astype(str).to_numpy()
        labels[:5] = "Rare"
        adata.obs["subclass"] = pd.Categorical(labels)
        adata.obs["class"] = pd.Categorical(np.where(labels == "Rare", "Rare", adata.obs["class"].astype(str)))

        sct.tl.detect_lineage_mixing(adata, lineage_key="class", use_raw=False, n_markers=20, min_group_size=20)

        rare = adata.obs["subclass"] == "Rare"
        self.assertTrue(adata.obs.loc[rare, "mixing_UMI_frac"].isna().all())
        self.assertEqual(int(adata.obs.loc[rare, "mixing_flag"].sum()), 0)

    def test_rejects_normalized_matrix(self):
        adata = _make_adata(n_pure=30, n_doublet=0, n_ambient=0, n_opc_oligo=0)
        adata.X = adata.X.multiply(1.0 / adata.X.sum(1)).tocsr() * 1e4
        adata.X.data = np.log1p(adata.X.data)

        with self.assertRaises(ValueError):
            sct.tl.detect_lineage_mixing(adata, lineage_key="class", use_raw=False)


    def test_known_lineage_sets_rejects_unknown_labels(self):
        adata = _make_adata(n_pure=30, n_doublet=0, n_ambient=0, n_opc_oligo=0)

        with self.assertRaises(KeyError):
            sct.tl.detect_lineage_mixing(
                adata, lineage_key="class", known_lineage_sets=[("OPC", "Oligodendrocyte")], use_raw=False
            )

    def test_results_write_to_h5ad(self):
        adata = _make_adata(n_pure=40, n_doublet=10, n_ambient=10, n_opc_oligo=0)

        sct.tl.detect_lineage_mixing(adata, use_raw=False, n_markers=20, library_key="channel")

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mixing.h5ad")
            adata.write_h5ad(path)
            back = ad.read_h5ad(path)
        self.assertIn("mixing_source", back.obs)
        self.assertIn("library_summary", back.uns["mixing"])

if __name__ == "__main__":
    unittest.main()
