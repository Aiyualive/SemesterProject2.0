import subprocess

def makeDfImage(df, index):
    """
    Makes dataframe into image.
    params:
        df: dataframe
        index: specifies how many rows to be converted
    """
    # Could also not include the css
    css = """
            <style type=\"text/css\">
            table {
            color: #333;
            font-family: Helvetica, Arial, sans-serif;
            width: 640px;
            border-collapse:
            collapse;
            border-spacing: 0;
            }
            td, th {
            border: 1px solid transparent; /* No more visible border */
            height: 30px;
            }
            th {
            background: #DFDFDF; /* Darken header a bit */
            font-weight: bold;
            }
            td {
            background: #FAFAFA;
            text-align: center;
            }
            table tr:nth-child(odd) td{
            background-color: white;
            }
            </style>
            """
    df = df.iloc[:index,:]
    text_file = open("table.html", "a")

    # write the CSS
    text_file.write(css)
    text_file.write(df.to_html())
    text_file.close()

    # Converting dataframe to png
    subprocess.run(
        f'wkhtmltoimage -f png --width 0 table.html table.png', shell=True)

def makeImages(d_df):
    """
    Stacks the 4 axle channels into 1 image
    params:
        d_df: dataframe that contains the defects
        dictionary: specifies the count of each defect type
    obs:
        requires imgs/concat folder
        works in unison with plot_defects.
    """

    defect_counts = d_df['def_type'].value_counts().to_dict()

    #divide by 4 since there are 4 axles
    dictionary = {k: v / 4 for k, v in defect_counts.iteritems()}

    for d_type, m in dictionary.items():
        for i in range(m):
            subprocess.run(
                f"""convert \
                imgs/{d_type}_{i}_AXLE_11.png \
                imgs/{d_type}_{i}_AXLE_12.png \
                imgs/{d_type}_{i}_AXLE_41.png \
                imgs/{d_type}_{i}_AXLE_42.png \
                -append ./imgs/concat/{d_type}_{i}.png""", shell=True)
