"""Check the tokenizer assumptions required by the Qwen3 EOS correction."""

from __future__ import annotations


def check_model_pair(student_path: str, teacher_path: str) -> None:
    from transformers import AutoTokenizer

    student = AutoTokenizer.from_pretrained(student_path)
    teacher = AutoTokenizer.from_pretrained(teacher_path)
    if student.get_vocab() != teacher.get_vocab():
        raise ValueError("LSPD requires identical student and teacher vocabularies.")
    if student.eos_token_id != 151643 or teacher.eos_token_id != 151645:
        raise ValueError("This recipe requires student EOS 151643 and teacher EOS 151645.")
    messages = [{"role": "user", "content": "Return the answer in \\boxed{}."}]
    kwargs = dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if student.apply_chat_template(messages, **kwargs) != teacher.apply_chat_template(messages, **kwargs):
        raise ValueError("Student and teacher must use the same non-thinking chat template.")
    print("Tokenizer preflight passed: shared vocabulary, non-thinking template, EOS correction 151645 -> 151643.")
