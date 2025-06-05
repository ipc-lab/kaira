from typing import Dict, Tuple

import requests
import torch

# Citation for the RPTU channel codes database
CITATION = "Michael Helmling, Stefan Scholl, Florian Gensheimer, Tobias Dietz, Kira Kraft, Oliver Griebel, Stefan Ruzika, and Norbert Wehn. Database of Channel Codes and ML Simulation Results. rptu.de/channel-codes, 2025."

EXISTING_CODES: Dict[Tuple[int, int], Dict[str, str]] = {
    (576, 288): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_576_0.5.alist"},
    (576, 384): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_576_0.66B.alist"},
    (576, 432): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_576_0.75A.alist", "wimaxB": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_576_0.75B.alist"},
    (576, 480): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_576_0.83.alist"},
    (672, 336): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_672_0.5.alist", "wigig": "https://rptu.de/fileadmin/chaco/public/alists_11ad/wigig_R05_N672_K336.alist"},
    (672, 420): {"wigig": "https://rptu.de/fileadmin/chaco/public/alists_11ad/wigig_R063_N672_K420.alist"},
    (672, 448): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_672_0.66B.alist"},
    (672, 504): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_672_0.75A.alist", "wimaxB": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_672_0.75B.alist", "wigig": "https://rptu.de/fileadmin/chaco/public/alists_11ad/wigig_R075_N672_K504.alist"},
    (672, 560): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_672_0.83.alist"},
    (672, 546): {"wigig": "https://rptu.de/fileadmin/chaco/public/alists_11ad/ieee_802_11ad_p42_n672_r081.alist"},
    (768, 384): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_768_0.5.alist"},
    (768, 512): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_768_0.66B.alist"},
    (768, 576): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_768_0.75A.alist", "wimaxB": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_768_0.75B.alist"},
    (768, 640): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_768_0.83.alist"},
    (864, 432): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_864_0.5.alist"},
    (864, 576): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_864_0.66B.alist"},
    (864, 648): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_864_0.75A.alist", "wimaxB": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_864_0.75B.alist"},
    (864, 720): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_864_0.83.alist"},
    (960, 480): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_960_0.5.alist"},
    (960, 640): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_960_0.66B.alist"},
    (960, 720): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_960_0.75A.alist", "wimaxB": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_960_0.75B.alist"},
    (960, 800): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_960_0.83.alist"},
    (1056, 528): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_1056_0.5.alist"},
    (1056, 704): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_1056_0.66B.alist"},
    (1056, 792): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_1056_0.75A.alist", "wimaxB": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_1056_0.75B.alist"},
    (1056, 880): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_1056_0.83.alist"},
    (1152, 576): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_1152_0.5.alist"},
    (1152, 768): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_1152_0.66B.alist"},
    (1152, 864): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_1152_0.75A.alist", "wimaxB": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_1152_0.75B.alist"},
    (1152, 960): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_1152_0.83.alist"},
    (1248, 624): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_1248_0.5.alist"},
    (1248, 832): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_1248_0.66B.alist"},
    (1248, 936): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_1248_0.75A.alist", "wimaxB": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_1248_0.75B.alist"},
    (1248, 1040): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_1248_0.83.alist"},
    (1344, 672): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_1344_0.5.alist"},
    (1344, 896): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_1344_0.66B.alist"},
    (1344, 1008): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_1344_0.75A.alist", "wimaxB": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_1344_0.75B.alist"},
    (1344, 1120): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_1344_0.83.alist"},
    (1440, 720): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_1440_0.5.alist"},
    (1440, 960): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_1440_0.66B.alist"},
    (1440, 1080): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_1440_0.75A.alist", "wimaxB": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_1440_0.75B.alist"},
    (1440, 1200): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_1440_0.83.alist"},
    (1536, 768): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_1536_0.5.alist"},
    (1536, 1024): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_1536_0.66B.alist"},
    (1536, 1152): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_1536_0.75A.alist", "wimaxB": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_1536_0.75B.alist"},
    (1536, 1280): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_1536_0.83.alist"},
    (1632, 816): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_1632_0.5.alist"},
    (1632, 1088): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_1632_0.66B.alist"},
    (1632, 1224): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_1632_0.75A.alist", "wimaxB": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_1632_0.75B.alist"},
    (1632, 1360): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_1632_0.83.alist"},
    (1728, 864): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_1728_0.5.alist"},
    (1728, 1152): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_1728_0.66B.alist"},
    (1728, 1296): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_1728_0.75A.alist", "wimaxB": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_1728_0.75B.alist"},
    (1728, 1440): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_1728_0.83.alist"},
    (1824, 912): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_1824_0.5.alist"},
    (1824, 1216): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_1824_0.66B.alist"},
    (1824, 1368): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_1824_0.75A.alist", "wimaxB": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_1824_0.75B.alist"},
    (1824, 1520): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_1824_0.83.alist"},
    (1920, 960): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_1920_0.5.alist"},
    (1920, 1280): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_1920_0.66B.alist"},
    (1920, 1440): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_1920_0.75A.alist", "wimaxB": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_1920_0.75B.alist"},
    (1920, 1600): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_1920_0.83.alist"},
    (2016, 1008): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_2016_0.5.alist"},
    (2016, 1344): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_2016_0.66B.alist"},
    (2016, 1512): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_2016_0.75A.alist", "wimaxB": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_2016_0.75B.alist"},
    (2016, 1680): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_2016_0.83.alist"},
    (2112, 1056): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_2112_0.5.alist"},
    (2112, 1408): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_2112_0.66B.alist"},
    (2112, 1584): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_2112_0.75A.alist", "wimaxB": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_2112_0.75B.alist"},
    (2112, 1760): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_2112_0.83.alist"},
    (2208, 1104): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_2208_0.5.alist"},
    (2208, 1472): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_2208_0.66B.alist"},
    (2208, 1656): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_2208_0.75A.alist", "wimaxB": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_2208_0.75B.alist"},
    (2208, 1840): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_2208_0.83.alist"},
    (2304, 1152): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_2304_0.5.alist"},
    (2304, 1536): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_2304_0.66B.alist"},
    (2304, 1728): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_2304_0.75A.alist", "wimaxB": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_2304_0.75B.alist"},
    (2304, 1920): {"wimax": "https://rptu.de/fileadmin/chaco/public/alists_wimax/wimax_2304_0.83.alist"},
    (648, 540): {"wifi": "https://rptu.de/fileadmin/chaco/public/alists_wifi/wifi_648_r083.alist"},
    (336, 168): {"itu_g.h": "https://rptu.de/fileadmin/chaco/public/alists_g.h/LDPC_N336_K196_ITU_G.h.alist"},
    (384, 192): {"wran": "https://rptu.de/fileadmin/chaco/public/alists_wran/WRAN_N384_K192_P16_R05.txt"},
    (384, 256): {"wran": "https://rptu.de/fileadmin/chaco/public/alists_wran/WRAN_N384_K256_P16_R066.txt"},
    (384, 288): {"wran": "https://rptu.de/fileadmin/chaco/public/alists_wran/WRAN_N384_K288_P16_R075.txt"},
    (384, 320): {"wran": "https://rptu.de/fileadmin/chaco/public/alists_wran/WRAN_N384_K320_P16_R083.txt"},
    (480, 240): {"wran": "https://rptu.de/fileadmin/chaco/public/alists_wran/WRAN_N480_K240_P20_R05.txt"},
    (480, 320): {"wran": "https://rptu.de/fileadmin/chaco/public/alists_wran/WRAN_N480_K320_P20_R066.txt"},
    (480, 360): {"wran": "https://rptu.de/fileadmin/chaco/public/alists_wran/WRAN_N480_K360_P20_R075.txt"},
    (480, 400): {"wran": "https://rptu.de/fileadmin/chaco/public/alists_wran/WRAN_N480_K400_P20_R083.txt"},
    (128, 64): {"ccsds": "https://rptu.de/fileadmin/chaco/public/alists_ccsds/CCSDS_ldpc_n128_k64.alist"},
    (256, 128): {"ccsds": "https://rptu.de/fileadmin/chaco/public/alists_ccsds/CCSDS_ldpc_n256_k128.alist"},
    (512, 256): {"ccsds": "https://rptu.de/fileadmin/chaco/public/alists_ccsds/CCSDS_ldpc_n512_k256.alist"},
}


def get_code_from_database(url: str) -> str:
    """Download the content of a file from a given URL.

    Args:
        url (str): The URL of the file to download.

    Returns:
        str: The content of the file as a string.

    Raises:
        ValueError: If the URL is empty or None.
        requests.HTTPError: If the HTTP request returned an unsuccessful status code.
    """
    if not url or not isinstance(url, str):
        raise ValueError("No URL provided or invalid URL type.")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.text


def parse_alist(file_content: str) -> torch.Tensor:
    """Parse the content of an LDPC alist file and return the parity-check matrix H.

    The alist format describes a sparse binary matrix (parity-check matrix) as follows:
        - Line 0: n m (number of columns, number of rows)
        - Line 1: max_col_weight max_row_weight
        - Line 2: space-separated column weights (length n)
        - Line 3: space-separated row weights (length m)
        - Next n lines: for each column, the row indices (1-based) where ones appear (padded with zeros)
        - Next m lines: for each row, the column indices (1-based) where ones appear (padded with zeros)

    Args:
        file_content (str): The content of the alist file as a string.

    Returns:
        torch.Tensor: The parity-check matrix H of shape (m, n) as a torch tensor.

    Raises:
        ValueError: If the file format is invalid or contains malformed numbers.
        IndexError: If the file doesn't have enough lines.
    """
    lines = file_content.splitlines()

    # Check if we have enough lines for basic format
    if len(lines) < 4:
        raise IndexError("Insufficient lines in alist file")

    try:
        n, m = map(int, lines[0].split())
    except (ValueError, IndexError):
        raise ValueError("Invalid format for matrix dimensions")

    # Check if we have enough lines for the column indices
    if len(lines) < 4 + n:
        raise IndexError("Insufficient lines for column indices")

    # Skip unused weight information (lines 1-3)

    # The next n lines are the column indices
    col_indices = lines[4 : 4 + n]
    # The next m lines are the row indices
    row_indices = lines[4 + n : 4 + n + m]

    H = torch.zeros((m, n), dtype=torch.int64)

    # Process column indices
    for col, line in enumerate(col_indices):
        try:
            indices = list(map(int, line.split()))
        except ValueError:
            raise ValueError(f"Invalid number format in column {col}")

        for idx in indices:
            if idx > 0:  # Skip padding zeros
                row_idx = idx - 1  # Convert 1-based to 0-based
                if row_idx < m:  # Only use valid row indices
                    H[row_idx, col] = 1

    # Process row indices for verification and completeness
    for row, line in enumerate(row_indices):
        try:
            indices = list(map(int, line.split()))
        except ValueError:
            raise ValueError(f"Invalid number format in row {row}")

        for idx in indices:
            if idx > 0:  # Skip padding zeros
                col_idx = idx - 1  # Convert 1-based to 0-based
                if col_idx < n:  # Only use valid column indices
                    if H[row, col_idx] != 1:
                        raise ValueError(f"Row {row} references column {col_idx} which is not set to 1")

    return H
