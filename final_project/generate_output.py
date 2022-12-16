import html
from pathlib import Path


def print_heading(text_html: str, size: str, output_html: html):
    """Write to html file with specified heading"""

    output_html.write(
        "<h1><center></center></h1>"
        f"<{size}><center>{text_html}</center></{size}>"
        "<h1><center></center></h1>"
    )

    return


def check_file(file_path: str, var_type: str, output_html: html):
    """Check files and write it to html file"""

    my_file = Path(f"{file_path}.html")

    if my_file.is_file():
        with open(f"{file_path}.html", "r", encoding="utf-8") as f:
            scores = f.read()
        print_heading(f"{var_type}", "h2", output_html)
        output_html.write("<body><center>%s</center></body>" % scores)
    # else:
    #     print_heading(f"{var_type}", "h2", output_html)
    #     output_html.write(f"<h3><center>{var_type} Not Present</center></h3>")

    return


def just_do_it():
    """creating a html file for output"""

    output = open("./files/baseball.html", "w")
    print_heading("Base Ball Data Statistics", "h1", output)

    # Write Response and Predictor Analysis file to final html
    check_file("./files/scores", "Response and Predictor Analysis", output)

    # Write Brute Force Analysis files to final html
    check_file(
        "./files/cont_cont_corr_table",
        "Continuous/Continuous Pairs Correlation Table",
        output,
    )

    check_file(
        "./files/cont_cont_corr_matrix",
        "Continuous/Continuous Pairs Correlation Matrix",
        output,
    )
    check_file(
        "./files/cont_cont_mean_table",
        "Continuous/Continuous Brute Force Table",
        output,
    )

    check_file(
        "./files/catg_cont_corr_table",
        "Category/Continuous Pairs Correlation Table",
        output,
    )
    check_file(
        "./files/catg_cont_corr_matrix",
        "Category/Continuous Pairs Correlation Matrix",
        output,
    )
    check_file(
        "./files/catg_cont_mean_table", "Category/Continuous Brute Force Table", output
    )

    check_file(
        "./files/catg_catg_corr_table",
        "Category/Category Pairs Correlation Table",
        output,
    )
    check_file(
        "./files/catg_catg_corr_matrix", "Category/Category Correlation Matrix", output
    )
    check_file(
        "./files/catg_catg_mean_table", "Category/Category Brute Force Table", output
    )

    # Write Model Statistics file to final html
    check_file("./files/models_stats", "ML Models Statistics", output)
    check_file("./files/plots/models_stats_roc", "ROC Curve", output)

    check_file(
        "./files/models_stats_pt",
        "ML Models Statistics After Removing Features for p<0.05 & |t|>=[1.96",
        output,
    )
    check_file("./files/plots/models_stats_pt_roc", "ROC Curve", output)

    check_file(
        "./files/models_stats_corr",
        "ML Models Statistics After Removing Features for corr > 0.95",
        output,
    )
    check_file("./files/plots/models_stats_corr_roc", "ROC Curve", output)
    # print_heading(
    #     "Random forest model performs best with 53.35% Accuracy and balanced Precision and Recall score",
    #     "h2",
    #     output,
    # )
    # output.write('<link rel="stylesheet" type="text/css" href="table_style.css">')

    output.close()

    # webbrowser.open("baseball.html")

    return
