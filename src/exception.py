import sys


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)

        _, _, exc_tb = error_detail.exc_info()

        self.file_name = exc_tb.tb_frame.f_code.co_filename
        self.line_number = exc_tb.tb_lineno
        self.error_message = error_message

    def __str__(self):

        return (
            f"\nError occurred in Python script:\n"
            f"File Name : {self.file_name}\n"
            f"Line No   : {self.line_number}\n"
            f"Error     : {self.error_message}"
        )