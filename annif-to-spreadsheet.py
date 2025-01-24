#!/usr/bin/env python3

"""Export annif suggestions to a spreadsheet for qualitative evaluation"""

import argparse
import csv
import os
import random
import sys

import gspread
from gspread import Spreadsheet, Worksheet, WorksheetNotFound
from gspread.utils import a1_range_to_grid_range, rowcol_to_a1
from gspread_formatting import *

LIGHT_GREY_3 = Color.fromHex("#f3f3f3")
LIGHT_GREEN_3 = Color.fromHex("#d9ead3")
LIGHT_BLUE_3 = Color.fromHex("#cfe2f3")
LIGHT_RED_3 = Color.fromHex("#f4cccc")
LIGHT_YELLOW_3 = Color.fromHex("#fff2cc")
DARK_GREY_3 = Color.fromHex("#666666")
MEDIUM_GREY_3 = Color.fromHex("#999999")
LIGHT_PURPLE_3 = Color.fromHex("#d9d2e9")
LIGHT_ORANGE_3 = Color.fromHex("#fce5cd")

MAX_RESULTS = 5  # ignore beyond 5 results
MAX_ROWS = 1000
MAX_COLS = 20
ROW_SKIP = 6  # max suggestions, plus 1 for a gap

NUM_REGEX = re.compile(r"\d+")


def grid_range_to_column_letter(grid_range: str) -> str:
    return re.sub(r'\d+', '', grid_range).upper()


def get_file_data(dir: str):
    data = []
    for filename in os.listdir(dir):
        if not filename.endswith(".txt"):
            continue
        name = filename[:-len(".txt")]
        with open(os.path.join(dir, filename), "r") as tf:
            text = tf.read()
        preds = []
        for tool in projects:
            with open(os.path.join(dir, f"{name}.{tool}"), "r") as rf:
                reader = csv.reader(rf, delimiter="\t")
                preds.append([row[1] for row in reader][0:MAX_RESULTS])
        data.append((name, text, preds))
    return data


def get_cell_data(data, tools):
    # Tool titles
    tool_titles = [f"Tool {i + 1}" for i, _ in enumerate(tools)]
    top_header = ["Texts", None, None]
    for tool_title in tool_titles:
        top_header.extend([tool_title, None, None, None])
    header = ["#", "ID", "Text"] + ["Labels", "Good", "Poor", None] * len(tools) + ["Translate?", "Translation"]
    cell_data = [
        top_header,
        header,
    ]
    row_offset = 3
    for i, (name, text, tool_preds) in enumerate(data):
        item_offset = i + 1
        row = [str(item_offset), name, text]
        for pset in tool_preds:
            if not pset:
                row.extend([None, None, None, None])
            else:
                row.extend([pset[0], None, None, None])
        row.extend([None,
                    f"=IF({rowcol_to_a1(row_offset, len(row) + 1)}, GoogleTranslate(C{row_offset}, \"auto\", \"en\"), \"\")"])
        cell_data.append(row)

        for i in range(1, ROW_SKIP):
            row = [None, None, None]  # skip the first three columns
            for pset in tool_preds:
                try:
                    pred = pset[i]
                except IndexError:
                    pred = None
                row.extend([pred, None, None, None])
            row.extend([None, None, None])
            cell_data.append(row)
        row_offset += ROW_SKIP
    return cell_data


def create_worksheet(args, spreadsheet: Spreadsheet, tools: list[str], worksheet_name: str = "Evaluation"):
    data = get_file_data(args.dir)
    cell_data = get_cell_data(data, tools)

    sheet_num_rows = max(MAX_ROWS, len(data) * ROW_SKIP + 2)
    sheet_num_cols = max(MAX_COLS, 3 + (4 * len(tools) + 2))

    # get or create the worksheet...
    try:
        worksheet = spreadsheet.worksheet(worksheet_name)
        if args.replace:
            spreadsheet.del_worksheet(worksheet)
            worksheet = spreadsheet.add_worksheet(worksheet_name, cols=sheet_num_cols, rows=sheet_num_rows)
    except WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(worksheet_name, cols=sheet_num_cols, rows=sheet_num_rows)

    # Insert the data
    if not args.format:
        worksheet.update(range_name="A1", values=cell_data, raw=False)

    format_worksheet(spreadsheet, worksheet, data, cell_data, sheet_num_rows, tools)


def merge_header_rows(spreadsheet, worksheet, tools):
    merge_headers = ["A1:C1"]
    for i, _ in enumerate(tools):
        offset = (i * 4) + 4
        merge_range = rowcol_to_a1(1, offset) + ":" + rowcol_to_a1(1, offset + 2)
        merge_headers.append(merge_range)
    # Merge all the cells...
    spreadsheet.batch_update(
        dict(
            requests=[
                dict(
                    mergeCells=dict(
                        mergeType="MERGE_ROWS",
                        range=a1_range_to_grid_range(r, worksheet.id)
                    )
                ) for r in merge_headers
            ]
        )
    )


def calculate_ranges(data, tools):
    """Calculate ranges for various parts of the worksheet that need formatting."""

    full_ranges = []
    merge_ranges = []
    border_ranges = []
    label_ranges = []
    good_ranges = []
    poor_ranges = []
    toggle_translation_ranges = []
    translation_ranges = []

    row_offset = 3  # first row after headers
    check_translate_col_num = (3 + ((len(tools) * 4) + 1))
    translate_col_num = check_translate_col_num + 1

    for name, text, tool_preds in data:
        toggle_translation_addr = rowcol_to_a1(row_offset, check_translate_col_num)
        translation_addr = rowcol_to_a1(row_offset, translate_col_num)
        end_addr = rowcol_to_a1(row_offset + (ROW_SKIP - 1), translate_col_num)
        full_range = f"A{row_offset}:{end_addr}"
        merge_range = f"A{row_offset}:C{row_offset + (ROW_SKIP - 1)}"
        border_range = f"A{row_offset + (ROW_SKIP - 1)}:{end_addr}"

        full_ranges.append(full_range)
        merge_ranges.append(merge_range)
        border_ranges.append(border_range)
        toggle_translation_ranges.append(toggle_translation_addr)
        translation_ranges.append(translation_addr)

        for ii in range(0, ROW_SKIP):
            col_offset = 4
            for predictions in tool_preds:
                try:
                    _ = predictions[ii]
                    label_ranges.append(rowcol_to_a1(row_offset, col_offset))
                    good_ranges.append(rowcol_to_a1(row_offset, col_offset + 1))
                    poor_ranges.append(rowcol_to_a1(row_offset, col_offset + 2))
                except IndexError:
                    pass
                col_offset += 4
            row_offset += 1

    return (
        full_ranges,
        merge_ranges,
        border_ranges,
        label_ranges,
        good_ranges,
        poor_ranges,
        toggle_translation_ranges,
        translation_ranges
    )


def format_worksheet(spreadsheet, worksheet, data, cell_data, sheet_num_rows, tools):
    """Format the worksheet, merging cells, setting borders, etc."""
    merge_header_rows(spreadsheet, worksheet, tools)

    (
        full_ranges,
        merge_ranges,
        border_ranges,
        label_ranges,
        good_ranges,
        poor_ranges,
        toggle_translation_ranges,
        translation_ranges
    ) = calculate_ranges(data, tools)

    # Merge various cells...
    spreadsheet.batch_update(
        dict(
            requests=[
                dict(
                    mergeCells=dict(
                        mergeType="MERGE_COLUMNS",
                        range=a1_range_to_grid_range(r, worksheet.id)
                    )
                ) for r in merge_ranges
            ]
        )
    )

    # Wrap all the text
    text_fmt = CellFormat(
        wrapStrategy="WRAP",
        verticalAlignment="TOP"
    )
    format_cell_ranges(worksheet, [(cell_range, text_fmt) for cell_range in full_ranges])

    # Set the borders
    bottom_border = Borders(bottom=Border(style="SOLID_THICK", color=DARK_GREY_3, width=3))
    format_cell_range(worksheet, f"A2:A{sheet_num_rows}", CellFormat(textFormat=TextFormat(bold=True)))
    for i, _ in enumerate(tools):
        col_offset = (i * 4) + 4
        format_cell_range(worksheet, rowcol_to_a1(2, col_offset) + ":" + rowcol_to_a1(sheet_num_rows, col_offset),
                          CellFormat(textFormat=TextFormat(bold=True)))

    # Make scores and enable translation checkboxes
    set_data_validation_for_cell_ranges(worksheet, [(r, DataValidationRule(
        showCustomUi=True,
        condition=BooleanCondition("BOOLEAN")
    )) for r in (good_ranges + poor_ranges + toggle_translation_ranges)])

    # set label colors to a light grey
    format_cell_ranges(worksheet, [(r, CellFormat(backgroundColor=LIGHT_GREY_3)) for r in label_ranges])
    format_cell_ranges(worksheet, [(r, CellFormat(backgroundColor=LIGHT_GREEN_3)) for r in good_ranges])
    format_cell_ranges(worksheet, [(r, CellFormat(backgroundColor=LIGHT_RED_3)) for r in poor_ranges])
    format_cell_ranges(worksheet, [(r, CellFormat(backgroundColor=LIGHT_PURPLE_3)) for r in toggle_translation_ranges])
    format_cell_ranges(worksheet, [(cell_range, text_fmt) for cell_range in translation_ranges])

    # Colour cells
    format_cell_range(worksheet, f"A1:A{len(cell_data)}",
                      CellFormat(backgroundColor=LIGHT_ORANGE_3, borders=bottom_border))
    format_cell_range(worksheet, f"B1:B{len(cell_data)}",
                      CellFormat(backgroundColor=LIGHT_GREEN_3, borders=bottom_border))
    format_cell_range(worksheet, f"C1:C{len(cell_data)}",
                      CellFormat(backgroundColor=LIGHT_BLUE_3, borders=bottom_border))
    format_cell_range(worksheet, f"A1:{rowcol_to_a1(1, len(cell_data[0]))}",
                      CellFormat(textFormat=TextFormat(bold=True, fontSize=18), backgroundColor=LIGHT_GREY_3))
    # Set widths of columns:
    col_widths = [("A", 50), ("B", 150), ("C", 800)]

    # Set spacer columns between tool suggestions narrower widths
    for i, _ in enumerate(tools):
        col_offset = (i * 4) + 7
        addr = rowcol_to_a1(1, col_offset)
        col_widths.append((grid_range_to_column_letter(addr), 50))

    # Increase the width of the translation value column a bit...
    col_widths.append((grid_range_to_column_letter(toggle_translation_ranges[0]), 80))
    col_widths.append((grid_range_to_column_letter(translation_ranges[0]), 300))

    set_column_widths(worksheet, col_widths)
    worksheet.rows_auto_resize(0, sheet_num_rows)
    # Add a bottom border to all the ranges:
    format_cell_ranges(worksheet, [(cell_range, CellFormat(borders=bottom_border)) for cell_range in border_ranges])
    # Add a left border to all the tool columns
    for i, _ in enumerate(tools):
        col_offset = (i * 4) + 4
        start_addr = rowcol_to_a1(1, col_offset)
        end_addr = rowcol_to_a1(sheet_num_rows, col_offset)
        format_cell_range(worksheet, f"{start_addr}:{end_addr}",
                          CellFormat(borders=Borders(left=Border(style="SOLID_MEDIUM", color=MEDIUM_GREY_3, width=3))))
    # Bold and freeze header
    format_cell_range(worksheet, "1:2", CellFormat(textFormat=TextFormat(bold=True)))
    worksheet.freeze(rows=2)

    # Protect all the text and label ranges, if an email is provided in the EDITOR_EMAIL environment variable:
    # We don't protect the text ranges since this prevents people from expanding the text to read it
    editor_email = os.environ.get("EDITOR_EMAIL")
    if editor_email:
        for i, _ in enumerate(tools):
            col_offset = (i * 4) + 4
            start_addr = rowcol_to_a1(1, col_offset)
            end_addr = rowcol_to_a1(sheet_num_rows, col_offset)
            worksheet.add_protected_range(f"{start_addr}:{end_addr}", editor_email)


def add_instructions(args, sheet):
    """Insert some instructions onto the first page of the sheet"""
    wks = sheet.sheet1
    wks.update_title("Instructions")

    instructions = [
        ["EHRI Term Suggestion Evaluation"],
        [],
        ["Instructions"],
        [],
        ["1.", "Read the text in the 'Text' column. If you can't see it all, "
               "double-click the cell to expand it."],
        ["2.",
         "For each label in the 'Labels' column, check the 'Good' box if "
         "it is a good label for the text, or the 'Poor' box if it is not."],
        ["3.", "If you are unsure, leave both boxes unchecked."],
        [],
        ["Translating text"],
        [
            "If the text is in an unfamiliar language you can use can translate it by "
            "clicking the checkbox in column H. "],
    ]

    wks.update(range_name="A1", values=instructions)

    # Format the instructions!
    format_cell_range(wks, "A1", CellFormat(textFormat=TextFormat(bold=True, fontSize=24)))
    format_cell_range(wks, "A3", CellFormat(textFormat=TextFormat(bold=True, fontSize=16)))
    format_cell_range(wks, "A9", CellFormat(textFormat=TextFormat(bold=True, fontSize=16)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export annif suggestions to a spreadsheet")
    parser.add_argument("dir", help="""The directory containing the suggestions,
                                       as generated by the Annif 'index' command,
                                       with a specific suffix for each tool under evaluation.
                                       These suffixes are referenced in the multiple 'project' arguments""")
    parser.add_argument("project", nargs="+", help="The Annif 'tool' name, to be found on the index file's suffix")
    parser.add_argument("-n", "--name", default="suggestions", help="The name of the spreadsheet")
    parser.add_argument("--replace", action="store_true", help="Replace the worksheet if it already exists")
    parser.add_argument('--no-randomise', action='store_true', help="Don't randomise the order of the tools")
    parser.add_argument("-f", "--format", action="store_true", help="Only format the spreadsheet, no data updates")

    args = parser.parse_args()

    if args.format and args.replace:
        print("Cannot replace and format at the same time", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(".secrets/credentials.json"):
        print("Error: Please provide the Google Sheets API credentials in the .secrets directory,"
                 "as per the 'gspread' package's instructions: https://docs.gspread.org/en/v6.0.0/oauth2.html", file=sys.stderr)
        sys.exit(1)

    credentials_filename = os.path.join(os.path.dirname(__file__), ".secrets", "credentials.json")
    authorized_user_filename = os.path.join(os.path.dirname(__file__), ".secrets", "authorized_user.json")
    gc = gspread.oauth(credentials_filename=credentials_filename, authorized_user_filename=authorized_user_filename)
    try:
        sheet = gc.open(args.name)
    except gspread.SpreadsheetNotFound:
        sheet = gc.create(args.name)
    print("Spreadsheet URL:", sheet.url)

    add_instructions(args, sheet)

    # Randomise the order of the tools

    projects = args.project
    if not args.no_randomise:
        random.shuffle(projects)
    create_worksheet(args, sheet, projects)
