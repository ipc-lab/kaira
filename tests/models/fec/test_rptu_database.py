"""Tests for the rptu_database module in kaira.models.fec package."""

from unittest.mock import Mock, patch

import pytest
import torch

from kaira.models.fec.rptu_database import (
    CITATION,
    EXISTING_CODES,
    get_code_from_database,
    parse_alist,
)


class TestRPTUDatabase:
    """Test suite for RPTU database functionality."""

    def test_citation_exists(self):
        """Test that citation is properly defined."""
        assert isinstance(CITATION, str)
        assert len(CITATION) > 0
        assert "RPTU" in CITATION or "rptu" in CITATION

    def test_existing_codes_structure(self):
        """Test the structure of EXISTING_CODES dictionary."""
        assert isinstance(EXISTING_CODES, dict)
        assert len(EXISTING_CODES) > 0

        # Check structure of dictionary
        for key, value in EXISTING_CODES.items():
            # Key should be tuple of two integers (n, k)
            assert isinstance(key, tuple)
            assert len(key) == 2
            assert isinstance(key[0], int)  # code length
            assert isinstance(key[1], int)  # code dimension
            assert key[0] > key[1]  # n > k

            # Value should be dict with string keys and URL values
            assert isinstance(value, dict)
            for sub_key, url in value.items():
                assert isinstance(sub_key, str)
                assert isinstance(url, str)
                assert url.startswith("https://")

    def test_existing_codes_specific_entries(self):
        """Test some specific entries in EXISTING_CODES."""
        # Test that some known codes exist
        assert (128, 64) in EXISTING_CODES
        assert (256, 128) in EXISTING_CODES
        assert (512, 256) in EXISTING_CODES

        # Test specific code types
        if (576, 288) in EXISTING_CODES:
            assert "wimax" in EXISTING_CODES[(576, 288)]

        if (672, 336) in EXISTING_CODES:
            codes_672_336 = EXISTING_CODES[(672, 336)]
            assert isinstance(codes_672_336, dict)

    @patch("requests.get")
    def test_get_code_from_database_success(self, mock_get):
        """Test successful download from database."""
        # Mock successful response
        mock_response = Mock()
        mock_response.text = "test content"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        url = "https://example.com/test.alist"
        result = get_code_from_database(url)

        assert result == "test content"
        mock_get.assert_called_once_with(url, timeout=30)
        mock_response.raise_for_status.assert_called_once()

    @patch("requests.get")
    def test_get_code_from_database_http_error(self, mock_get):
        """Test HTTP error handling."""
        # Mock HTTP error
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("HTTP Error")
        mock_get.return_value = mock_response

        url = "https://example.com/test.alist"

        with pytest.raises(Exception, match="HTTP Error"):
            get_code_from_database(url)

    def test_get_code_from_database_invalid_url(self):
        """Test invalid URL handling."""
        # Test empty URL
        with pytest.raises(ValueError, match="No URL provided"):
            get_code_from_database("")

        # Test None URL
        with pytest.raises(ValueError, match="No URL provided"):
            get_code_from_database(None)

        # Test non-string URL
        with pytest.raises(ValueError, match="invalid URL type"):
            get_code_from_database(123)

    def test_parse_alist_basic(self):
        """Test basic alist parsing functionality."""
        # Create a simple alist file content
        alist_content = """3 2
3 2
2 2 1
2 1
1 2
1 3
2
1 2
3"""

        H = parse_alist(alist_content)

        assert H.shape == (2, 3)
        assert H.dtype == torch.int64

        # Check specific values
        expected = torch.tensor([[1, 1, 0], [1, 0, 1]], dtype=torch.int64)
        assert torch.equal(H, expected)

    def test_parse_alist_larger_matrix(self):
        """Test parsing of larger alist matrix."""
        # Create a simple consistent 4x6 matrix
        # Target matrix: [[1,1,0,0,0,0], [1,0,1,0,0,0], [0,1,0,1,0,0], [0,0,1,1,0,0]]
        alist_content = """6 4
2 2
2 2 2 2 0 0
2 2 2 2
1 2
1 3
2 4
3 4
0 0
0 0
1 2
1 3
2 4
3 4"""

        H = parse_alist(alist_content)

        assert H.shape == (4, 6)
        assert H.dtype == torch.int64

        # Check that matrix is binary
        assert torch.all((H == 0) | (H == 1))

        # Based on the column indices created above
        expected = torch.tensor([[1, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0], [0, 1, 0, 1, 0, 0], [0, 0, 1, 1, 0, 0]], dtype=torch.int64)
        assert torch.equal(H, expected)

    def test_parse_alist_with_padding_zeros(self):
        """Test alist parsing with padding zeros."""
        # Alist format often pads with zeros
        alist_content = """4 2
2 2
2 2 0 0
2 2
1 2 0 0
1 2 0 0
0 0 0 0
0 0 0 0
1 2 0 0
1 2 0 0"""

        H = parse_alist(alist_content)

        assert H.shape == (2, 4)

        # Check that padding zeros are ignored
        expected = torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=torch.int64)
        assert torch.equal(H, expected)

    def test_parse_alist_single_row(self):
        """Test alist parsing with single row."""
        alist_content = """3 1
1 3
1 1 1
3
1
1
1
1 2 3"""

        H = parse_alist(alist_content)

        assert H.shape == (1, 3)
        expected = torch.tensor([[1, 1, 1]], dtype=torch.int64)
        assert torch.equal(H, expected)

    def test_parse_alist_single_column(self):
        """Test alist parsing with single column."""
        alist_content = """1 3
3 1
3
1 1 1
1 2 3
1
1
1"""

        H = parse_alist(alist_content)

        assert H.shape == (3, 1)
        expected = torch.tensor([[1], [1], [1]], dtype=torch.int64)
        assert torch.equal(H, expected)

    def test_parse_alist_sparse_matrix(self):
        """Test alist parsing with very sparse matrix."""
        alist_content = """5 3
1 2
1 1 1 1 1
2 1 1
1 0 0 0 0
2 0 0 0 0
3 0 0 0 0
0 0 0 0 0
0 0 0 0 0
1
2
3"""

        H = parse_alist(alist_content)

        assert H.shape == (3, 5)

        # Check that it's a diagonal-like sparse matrix
        expected = torch.tensor([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]], dtype=torch.int64)
        assert torch.equal(H, expected)

    def test_parse_alist_empty_rows(self):
        """Test alist parsing with some empty positions."""
        alist_content = """3 2
1 2
1 1 1
2 1
1 0
2 0
2 0
1
2 3"""
        H = parse_alist(alist_content)

        assert H.shape == (2, 3)
        expected = torch.tensor([[1, 0, 0], [0, 1, 1]], dtype=torch.int64)
        assert torch.equal(H, expected)

    def test_parse_alist_invalid_format(self):
        """Test alist parsing with invalid format."""
        # Test with insufficient lines
        invalid_content = """3 2
2 2"""

        with pytest.raises(IndexError):
            parse_alist(invalid_content)

    def test_parse_alist_malformed_numbers(self):
        """Test alist parsing with malformed numbers."""
        # Test with non-integer values
        invalid_content = """3.5 2
2 2
2 2 1
2 1
1 2
1 3
2
1 2 3
1 3"""

        with pytest.raises(ValueError):
            parse_alist(invalid_content)

    def test_code_parameters_consistency(self):
        """Test that code parameters in EXISTING_CODES are consistent."""
        for (n, k), codes in EXISTING_CODES.items():
            # Check that n > k (valid code rate)
            assert n > k, f"Invalid code parameters: n={n}, k={k}"

            # Check that k > 0 and n > 0
            assert k > 0, f"Invalid code dimension: k={k}"
            assert n > 0, f"Invalid code length: n={n}"

            # Check that codes dict is not empty
            assert len(codes) > 0, f"No codes defined for parameters ({n}, {k})"

    def test_url_format_consistency(self):
        """Test that all URLs in EXISTING_CODES have consistent format."""
        for codes in EXISTING_CODES.values():
            for code_type, url in codes.items():
                assert url.startswith("https://rptu.de/"), f"Invalid URL format: {url}"
                assert code_type in ["wimax", "wimaxB", "wigig", "wifi", "itu_g.h", "wran", "ccsds"], f"Unknown code type: {code_type}"

    def test_integration_with_mock_data(self):
        """Test integration of download and parse functions."""
        # Mock alist content
        mock_alist = """4 2
2 2
2 2 2 2
2 2
1 2 0 0
3 4 0 0
1 2 3 4
1 2 3 4"""

        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.text = mock_alist
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            # Download and parse
            url = "https://example.com/test.alist"
            content = get_code_from_database(url)
            H = parse_alist(content)

            assert H.shape == (2, 4)
            assert torch.all((H == 0) | (H == 1))

    def test_real_code_parameters(self):
        """Test some real code parameters from the database."""
        # Test CCSDS codes
        if (128, 64) in EXISTING_CODES:
            assert "ccsds" in EXISTING_CODES[(128, 64)]

        # Test WiMAX codes
        if (576, 288) in EXISTING_CODES:
            assert "wimax" in EXISTING_CODES[(576, 288)]

        # Test code rates are reasonable
        for n, k in EXISTING_CODES.keys():
            rate = k / n
            assert 0 < rate < 1, f"Invalid code rate {rate} for ({n}, {k})"
            assert rate <= 0.95, f"Code rate too high: {rate} for ({n}, {k})"
