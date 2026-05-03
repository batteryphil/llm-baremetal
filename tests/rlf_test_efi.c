/* rlf_test_efi.c — Minimal UEFI test for RLF SymbolicPreprocessor
 * Only exercises rlf_preprocess() + rlf_ftoa() — no backbone inference.
 * This keeps BSS under 64KB so UEFI loads it without page faults. */
#include <efi.h>
#include <efilib.h>

/* Pull in only the types we need from rlf_engine.h */
#include "../../engine/ssm/rlf_engine.h"

/* repl_print stub — write ASCII to EFI console */
void repl_print(const char *s) {
    if (!s) return;
    while (*s) {
        CHAR16 ch[2] = { (CHAR16)(unsigned char)*s++, 0 };
        ST->ConOut->OutputString(ST->ConOut, ch);
    }
}

static void print16(const CHAR16 *s) {
    ST->ConOut->OutputString(ST->ConOut, (CHAR16*)s);
}

static void pline(const CHAR16 *s) {
    print16(s);
    print16(L"\r\n");
}

EFI_STATUS EFIAPI efi_main(EFI_HANDLE ImageHandle, EFI_SYSTEM_TABLE *SystemTable) {
    InitializeLib(ImageHandle, SystemTable);
    ST->ConOut->ClearScreen(ST->ConOut);

    pline(L"");
    pline(L"=====================================");
    pline(L"  RLF Engine Bare-Metal UEFI Test   ");
    pline(L"=====================================");
    pline(L"");

    /* Test 1: /rlf_status with NULL weights */
    pline(L"[TEST 1] rlf_status(NULL):");
    rlf_status_print(NULL);
    pline(L"");

    /* Test 2: SymbolicPreprocessor — alpha=47. beta=alpha. What is beta? */
    pline(L"[TEST 2] rlf_preprocess:");
    pline(L"  Input: alpha=47. beta=alpha. What is beta?");

    static RlfCtx ctx;
    rlf_preprocess("alpha=47. beta=alpha. What is beta?", &ctx);

    print16(L"  Result: ");
    repl_print(ctx.prompt_pp);
    pline(L"");

    int pass = 0;
    for (int i = 0; i < ctx.env.n; i++) {
        RlfBinding *b = &ctx.env.bindings[i];
        char nbuf[32];
        rlf_ftoa(b->numeric, nbuf);
        print16(L"  ");
        repl_print(b->name);
        print16(L" = ");
        if (b->resolved) {
            repl_print(nbuf);
            /* Check: alpha should resolve to 47 */
            if (b->name[0] == 'a' && b->numeric > 46.9f && b->numeric < 47.1f) pass++;
            /* Check: beta should resolve to 47 (via alpha) */
            if (b->name[0] == 'b' && b->numeric > 46.9f && b->numeric < 47.1f) pass++;
        } else {
            repl_print(b->value);
            print16(L" (unresolved)");
        }
        pline(L"");
    }

    pline(L"");
    if (pass >= 2) {
        pline(L"[TEST 2] PASSED - alpha=47, beta=47 correctly resolved");
    } else {
        pline(L"[TEST 2] FAIL - binding resolution incorrect");
    }

    /* Test 3: rlf_preprocess with arithmetic */
    pline(L"");
    pline(L"[TEST 3] rlf_preprocess arithmetic:");
    pline(L"  Input: x=10. y=x+5. z=y*2. What is z?");
    static RlfCtx ctx2;
    rlf_preprocess("x=10. y=x+5. z=y*2. What is z?", &ctx2);
    print16(L"  Result: ");
    repl_print(ctx2.prompt_pp);
    pline(L"");
    for (int i = 0; i < ctx2.env.n; i++) {
        RlfBinding *b = &ctx2.env.bindings[i];
        char nbuf[32];
        rlf_ftoa(b->numeric, nbuf);
        print16(L"  "); repl_print(b->name); print16(L" = ");
        repl_print(b->resolved ? nbuf : b->value);
        pline(b->resolved ? L"" : L" (unresolved)");
    }

    pline(L"");
    pline(L"=====================================");
    pline(L"  RLF preprocessor test complete    ");
    pline(L"=====================================");

    /* Wait for keypress then exit cleanly */
    EFI_INPUT_KEY key;
    ST->ConIn->Reset(ST->ConIn, FALSE);
    UINTN idx = 0;
    EFI_EVENT evts[1] = { ST->ConIn->WaitForKey };
    BS->WaitForEvent(1, evts, &idx);
    ST->ConIn->ReadKeyStroke(ST->ConIn, &key);

    return EFI_SUCCESS;
}
