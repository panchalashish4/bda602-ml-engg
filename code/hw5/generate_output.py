import html
import webbrowser
from pathlib import Path


def print_heading(text_html: str, size: str, output_html: html):
    output_html.write(
        "<h1><center></center></h1>"
        f"<{size}><center>{text_html}</center></{size}>"
        "<h1><center></center></h1>"
    )

    return


def check_file(file_path: str, var_type: str, output_html: html):
    my_file = Path(f"./plots/{file_path}.html")

    if my_file.is_file():
        with open(f"./plots/{file_path}.html", "r", encoding="utf-8") as f:
            scores = f.read()
        print_heading(f"{var_type}", "h2", output_html)
        output_html.write("<body><center>%s</center></body>" % scores)
    else:
        print_heading(f"{var_type}", "h2", output_html)
        output_html.write(f"<h3><center>{var_type} Not Present</center></h3>")

    return


def just_do_it():
    """creating a html file for output"""

    output = open("baseball.html", "w")

    print_heading("Base Ball Data Statistics", "h1", output)

    check_file("scores", "Response and Predictor Analysis", output)

    check_file(
        "cont_cont_corr_table", "Continuous/Continuous Pairs Correlation Table", output
    )
    check_file(
        "cont_cont_corr_matrix",
        "Continuous/Continuous Pairs Correlation Matrix",
        output,
    )
    check_file(
        "cont_cont_mean_table", "Continuous/Continuous Brute Force Table", output
    )

    check_file(
        "catg_cont_corr_table", "Category/Continuous Pairs Correlation Table", output
    )
    check_file(
        "catg_cont_corr_matrix", "Category/Continuous Pairs Correlation Matrix", output
    )
    check_file("catg_cont_mean_table", "Category/Continuous Brute Force Table", output)

    check_file(
        "catg_catg_corr_table", "Category/Category Pairs Correlation Table", output
    )
    check_file("catg_catg_corr_matrix", "Category/Category Correlation Matrix", output)
    check_file("catg_catg_mean_table", "Category/Category Brute Force Table", output)

    output.close()

    webbrowser.open("baseball.html")

    return
