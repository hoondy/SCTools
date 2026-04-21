"""Plotting helpers for SCTools."""

from ._shared import (
    _get_sample_size,
    _require_adjust_text,
    _require_matplotlib,
    _require_pegasus,
    _require_seaborn,
    defaultdict,
    np,
    pd,
    stats,
)


def scree_plot(data):
    """Plot the PCA scree curve from a Pegasus object."""
    plt, _ = _require_matplotlib()
    fig = plt.figure()
    plt.plot(range(1, data.uns["PCs"].shape[1] + 1), data.uns["pca"]["variance_ratio"], "o-", linewidth=2, color="blue")
    plt.title("Scree Plot")
    plt.xlabel("Principal Component")
    plt.ylabel("Variance Ratio")
    return fig


def corrFeatures(data, gene, top_n=50, sample_size=5000, vmin=0.0, vmax=1.0):
    """Plot a covariance-based local correlation map around one gene."""
    sns = _require_seaborn()
    _, rc_context = _require_matplotlib()

    sample_size = _get_sample_size(data.shape[0], sample_size)
    obs_indices = np.random.choice(data.shape[0], size=sample_size, replace=False)
    cov_mat = np.cov(data.X[obs_indices].T.todense())
    idx = data.var.index.get_loc(gene)
    top_cor_idx = np.argsort(cov_mat[idx])
    corr = cov_mat[top_cor_idx[-top_n:]][:, top_cor_idx[-top_n:]]
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True

    corr = pd.DataFrame(corr)
    corr.columns = data.var.iloc[top_cor_idx[-top_n:]].index.tolist()
    corr.index = data.var.iloc[top_cor_idx[-top_n:]].index.tolist()

    with rc_context({"figure.figsize": (12, 12)}):
        sns.set(style="white", font_scale=0.75)
        sns.heatmap(corr, mask=mask, vmin=vmin, vmax=vmax, square=True, cmap="YlGnBu", linewidth=0.1)
    return corr.index.tolist()


def corrPlot(corr, gene, top_n=50, vmin=-1.0, vmax=1.0, figsize=15, clustermap=False):
    """Plot a heatmap or clustermap for the strongest correlations to one gene."""
    sns = _require_seaborn()
    _, rc_context = _require_matplotlib()

    idx = corr.index.get_loc(gene)
    top_cor_idx = np.argsort(abs(corr.iloc[idx]))
    subset = corr.iloc[top_cor_idx[-top_n:], top_cor_idx[-top_n:]]

    with rc_context({"figure.figsize": (figsize, figsize)}):
        sns.set(style="white", font_scale=0.7)
        if clustermap:
            sns.clustermap(subset, vmin=vmin, vmax=vmax, cmap="vlag", linewidth=0.1)
        else:
            sns.heatmap(subset, vmin=vmin, vmax=vmax, cmap="vlag", linewidth=0.1, square=True)

    return subset


def corrFeatures2(data, gene, top_n=50, sample_size=1000, vmin=-1.0, vmax=1.0):
    """Compute and cluster a correlation submatrix centered on one gene."""
    sns = _require_seaborn()
    _, rc_context = _require_matplotlib()

    sample_size = _get_sample_size(data.shape[0], sample_size)
    obs_indices = np.random.choice(data.shape[0], size=sample_size, replace=False)

    exp_df = pd.DataFrame(data.X[obs_indices].todense())
    exp_df.columns = data.var_names
    exp_df = exp_df.loc[:, exp_df.var(axis=0) != 0].copy()

    corr = exp_df.corr()
    idx = corr.index.get_loc(gene)
    top_cor_idx = np.argsort(abs(corr.iloc[idx]))
    subset = corr.iloc[top_cor_idx[-top_n:], top_cor_idx[-top_n:]]

    with rc_context({"figure.figsize": (15, 15)}):
        sns.set(style="white", font_scale=0.75)
        sns.clustermap(subset, vmin=vmin, vmax=vmax, cmap="vlag", linewidth=0.1)

    return subset


def pal_max():
    """Return a large categorical color palette as a comma-separated string."""
    max_268 = ["#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059","#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87","#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80","#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100","#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F","#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09","#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66","#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C","#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81","#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00","#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700","#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329","#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C","#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800","#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51","#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58","#7A7BFF", "#D68E01", "#353339", "#78AFA1", "#FEB2C6", "#75797C", "#837393", "#943A4D","#B5F4FF", "#D2DCD5", "#9556BD", "#6A714A", "#001325", "#02525F", "#0AA3F7", "#E98176","#DBD5DD", "#5EBCD1", "#3D4F44", "#7E6405", "#02684E", "#962B75", "#8D8546", "#9695C5","#E773CE", "#D86A78", "#3E89BE", "#CA834E", "#518A87", "#5B113C", "#55813B", "#E704C4","#00005F", "#A97399", "#4B8160", "#59738A", "#FF5DA7", "#F7C9BF", "#643127", "#513A01","#6B94AA", "#51A058", "#A45B02", "#1D1702", "#E20027", "#E7AB63", "#4C6001", "#9C6966","#64547B", "#97979E", "#006A66", "#391406", "#F4D749", "#0045D2", "#006C31", "#DDB6D0","#7C6571", "#9FB2A4", "#00D891", "#15A08A", "#BC65E9", "#FFFFFE", "#C6DC99", "#203B3C","#671190", "#6B3A64", "#F5E1FF", "#FFA0F2", "#CCAA35", "#374527", "#8BB400", "#797868","#C6005A", "#3B000A", "#C86240", "#29607C", "#402334", "#7D5A44", "#CCB87C", "#B88183","#AA5199", "#B5D6C3", "#A38469", "#9F94F0", "#A74571", "#B894A6", "#71BB8C", "#00B433","#789EC9", "#6D80BA", "#953F00", "#5EFF03", "#E4FFFC", "#1BE177", "#BCB1E5", "#76912F","#003109", "#0060CD", "#D20096", "#895563", "#29201D", "#5B3213", "#A76F42", "#89412E","#1A3A2A", "#494B5A", "#A88C85", "#F4ABAA", "#A3F3AB", "#00C6C8", "#EA8B66", "#958A9F","#BDC9D2", "#9FA064", "#BE4700", "#658188", "#83A485", "#453C23", "#47675D", "#3A3F00","#061203", "#DFFB71", "#868E7E", "#98D058", "#6C8F7D", "#D7BFC2", "#3C3E6E", "#D83D66","#2F5D9B", "#6C5E46", "#D25B88", "#5B656C", "#00B57F", "#545C46", "#866097", "#365D25","#252F99", "#00CCFF", "#674E60", "#FC009C", "#92896B"]
    return ",".join(max_268)


class PySankeyException(Exception):
    pass


class NullsInFrame(PySankeyException):
    pass


class LabelMismatch(PySankeyException):
    pass


def check_data_matches_labels(labels, data, side):
    """Validate that user-supplied Sankey labels match the observed data."""
    if len(labels) > 0:
        if isinstance(data, list):
            data = set(data)
        if isinstance(data, pd.Series):
            data = set(data.unique().tolist())
        if isinstance(labels, list):
            labels = set(labels)
        if labels != data:
            msg = "\n"
            if len(labels) <= 20:
                msg = "Labels: " + ",".join(labels) + "\n"
            if len(data) < 20:
                msg += "Data: " + ",".join(data)
            raise LabelMismatch("{0} labels and data do not match.{1}".format(side, msg))


def sankey(
    left,
    right,
    leftWeight=None,
    rightWeight=None,
    colorDict=None,
    leftLabels=None,
    rightLabels=None,
    aspect=4,
    rightColor=False,
    fontsize=10,
    figureName=None,
    closePlot=False,
    size_x=6,
    size_y=12,
):
    """Create a Sankey diagram from left/right label assignments."""
    plt, _ = _require_matplotlib()
    sns = _require_seaborn()
    if leftWeight is None:
        leftWeight = []
    if rightWeight is None:
        rightWeight = []
    if leftLabels is None:
        leftLabels = []
    if rightLabels is None:
        rightLabels = []
    if len(leftWeight) == 0:
        leftWeight = np.ones(len(left))
    if len(rightWeight) == 0:
        rightWeight = leftWeight

    plt.figure()
    plt.rc("text", usetex=False)
    plt.rc("font", family="sans-serif")

    if isinstance(left, pd.Series):
        left = left.reset_index(drop=True)
    if isinstance(right, pd.Series):
        right = right.reset_index(drop=True)
    dataFrame = pd.DataFrame(
        {"left": left, "right": right, "leftWeight": leftWeight, "rightWeight": rightWeight},
        index=range(len(left)),
    )

    if len(dataFrame[(dataFrame.left.isnull()) | (dataFrame.right.isnull())]):
        raise NullsInFrame("Sankey graph does not support null values.")

    allLabels = pd.Series(np.r_[dataFrame.left.unique(), dataFrame.right.unique()]).unique()

    if len(leftLabels) == 0:
        leftLabels = pd.Series(dataFrame.left.unique()).unique()
    else:
        check_data_matches_labels(leftLabels, dataFrame["left"], "left")

    if len(rightLabels) == 0:
        rightLabels = pd.Series(dataFrame.right.unique()).unique()
    else:
        check_data_matches_labels(rightLabels, dataFrame["right"], "right")

    if colorDict is None:
        colorDict = {}
        colorPalette = sns.color_palette("hls", len(allLabels))
        for i, label in enumerate(allLabels):
            colorDict[label] = colorPalette[i]
    else:
        missing = [label for label in allLabels if label not in colorDict.keys()]
        if missing:
            msg = "The colorDict parameter is missing values for the following labels : "
            msg += "{}".format(", ".join(missing))
            raise ValueError(msg)

    ns_l = defaultdict()
    ns_r = defaultdict()
    for leftLabel in leftLabels:
        leftDict = {}
        rightDict = {}
        for rightLabel in rightLabels:
            leftDict[rightLabel] = dataFrame[(dataFrame.left == leftLabel) & (dataFrame.right == rightLabel)].leftWeight.sum()
            rightDict[rightLabel] = dataFrame[(dataFrame.left == leftLabel) & (dataFrame.right == rightLabel)].rightWeight.sum()
        ns_l[leftLabel] = leftDict
        ns_r[leftLabel] = rightDict

    leftWidths = defaultdict()
    topEdge = 0
    for i, leftLabel in enumerate(leftLabels):
        myD = {}
        myD["left"] = dataFrame[dataFrame.left == leftLabel].leftWeight.sum()
        if i == 0:
            myD["bottom"] = 0
            myD["top"] = myD["left"]
        else:
            myD["bottom"] = leftWidths[leftLabels[i - 1]]["top"] + 0.02 * dataFrame.leftWeight.sum()
            myD["top"] = myD["bottom"] + myD["left"]
            topEdge = myD["top"]
        leftWidths[leftLabel] = myD

    rightWidths = defaultdict()
    for i, rightLabel in enumerate(rightLabels):
        myD = {}
        myD["right"] = dataFrame[dataFrame.right == rightLabel].rightWeight.sum()
        if i == 0:
            myD["bottom"] = 0
            myD["top"] = myD["right"]
        else:
            myD["bottom"] = rightWidths[rightLabels[i - 1]]["top"] + 0.02 * dataFrame.rightWeight.sum()
            myD["top"] = myD["bottom"] + myD["right"]
            topEdge = myD["top"]
        rightWidths[rightLabel] = myD

    if topEdge == 0 and len(rightLabels) > 0:
        topEdge = rightWidths[rightLabels[0]]["top"]
    if topEdge == 0 and len(leftLabels) > 0:
        topEdge = leftWidths[leftLabels[0]]["top"]
    xMax = topEdge / aspect

    for leftLabel in leftLabels:
        plt.fill_between(
            [-0.02 * xMax, 0],
            2 * [leftWidths[leftLabel]["bottom"]],
            2 * [leftWidths[leftLabel]["bottom"] + leftWidths[leftLabel]["left"]],
            color=colorDict[leftLabel],
            alpha=0.99,
        )
        plt.text(
            -0.05 * xMax,
            leftWidths[leftLabel]["bottom"] + 0.5 * leftWidths[leftLabel]["left"],
            leftLabel,
            {"ha": "right", "va": "center"},
            fontsize=fontsize,
        )
    for rightLabel in rightLabels:
        plt.fill_between(
            [xMax, 1.02 * xMax],
            2 * [rightWidths[rightLabel]["bottom"]],
            2 * [rightWidths[rightLabel]["bottom"] + rightWidths[rightLabel]["right"]],
            color=colorDict[rightLabel],
            alpha=0.99,
        )
        plt.text(
            1.05 * xMax,
            rightWidths[rightLabel]["bottom"] + 0.5 * rightWidths[rightLabel]["right"],
            rightLabel,
            {"ha": "left", "va": "center"},
            fontsize=fontsize,
        )

    for leftLabel in leftLabels:
        for rightLabel in rightLabels:
            labelColor = rightLabel if rightColor else leftLabel
            if len(dataFrame[(dataFrame.left == leftLabel) & (dataFrame.right == rightLabel)]) > 0:
                ys_d = np.array(50 * [leftWidths[leftLabel]["bottom"]] + 50 * [rightWidths[rightLabel]["bottom"]])
                ys_d = np.convolve(ys_d, 0.05 * np.ones(20), mode="valid")
                ys_d = np.convolve(ys_d, 0.05 * np.ones(20), mode="valid")
                ys_u = np.array(
                    50 * [leftWidths[leftLabel]["bottom"] + ns_l[leftLabel][rightLabel]]
                    + 50 * [rightWidths[rightLabel]["bottom"] + ns_r[leftLabel][rightLabel]]
                )
                ys_u = np.convolve(ys_u, 0.05 * np.ones(20), mode="valid")
                ys_u = np.convolve(ys_u, 0.05 * np.ones(20), mode="valid")

                leftWidths[leftLabel]["bottom"] += ns_l[leftLabel][rightLabel]
                rightWidths[rightLabel]["bottom"] += ns_r[leftLabel][rightLabel]
                plt.fill_between(np.linspace(0, xMax, len(ys_d)), ys_d, ys_u, alpha=0.65, color=colorDict[labelColor])
    plt.gca().axis("off")
    plt.gcf().set_size_inches(size_x, size_y)
    if figureName is not None:
        plt.savefig("{}.png".format(figureName), bbox_inches="tight", dpi=150)
    if closePlot:
        plt.close()


def plot_correlation_circle(data, rep, features="highly_variable_features"):
    """Plot feature correlations against the first two dimensions of a representation."""
    pg = _require_pegasus()
    plt, _ = _require_matplotlib()
    adjust_text = _require_adjust_text()

    keyword = pg.select_features(data, features)
    tmp_X = data.uns[keyword]
    if features:
        feature_names = data.var[data.var[features]].index.tolist()
    else:
        feature_names = data.var.index.tolist()

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    texts = []
    for i in range(0, len(feature_names)):
        corr1, _ = stats.pearsonr(tmp_X[:, i], data.obsm["X_" + rep][:, 0])
        corr2, _ = stats.pearsonr(tmp_X[:, i], data.obsm["X_" + rep][:, 1])

        ax.arrow(0, 0, corr1, corr2, head_width=0.01, head_length=0.01, lw=0.1)

        if np.sqrt(corr1 ** 2 + corr2 ** 2) >= 0.25:
            texts.append(plt.text(corr1, corr2, feature_names[i], size=4))

    adjust_text(texts, arrowprops=dict(arrowstyle="-", color="black", lw=0.5))

    an = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(an), np.sin(an))
    plt.axis("equal")
    ax.set_title("Correlation circle plot of " + rep)
    plt.show()


__all__ = [
    "LabelMismatch",
    "NullsInFrame",
    "PySankeyException",
    "check_data_matches_labels",
    "corrFeatures",
    "corrFeatures2",
    "corrPlot",
    "pal_max",
    "plot_correlation_circle",
    "sankey",
    "scree_plot",
]
