# ── Makefile for Mamba2 SSM + RLF Bare-Metal UEFI ──
#
# Builds the complete llm-baremetal system with Mamba2 SSM inference engine.
# Based on llm-baremetal (Djiby Diop) + RLF (batteryphil)
#
# Usage:
#   make              — Build SSM UEFI application (llama2.efi)
#   make python-test  — Run Python verification suite
#   make clean        — Remove build artifacts

ARCH = x86_64
CC = gcc

# GNU-EFI paths (auto-detect multi-arch)
MULTIARCH ?= $(shell $(CC) -print-multiarch 2>/dev/null)
EFI_LIBDIR_CANDIDATES := /usr/lib /usr/lib/$(MULTIARCH)

EFI_LDS := $(firstword $(wildcard $(addsuffix /elf_$(ARCH)_efi.lds,$(EFI_LIBDIR_CANDIDATES))))
EFI_CRT0 := $(firstword $(wildcard $(addsuffix /crt0-efi-$(ARCH).o,$(EFI_LIBDIR_CANDIDATES))))
EFI_LIBDIR := $(firstword $(foreach d,$(EFI_LIBDIR_CANDIDATES),$(if $(wildcard $(d)/libgnuefi.a),$(d),)))

ifeq ($(strip $(EFI_LDS)),)
$(warning GNU-EFI not found: elf_$(ARCH)_efi.lds missing. UEFI build disabled.)
$(warning Install with: sudo apt install gnu-efi)
NO_UEFI = 1
endif
ifeq ($(strip $(EFI_CRT0)),)
$(warning GNU-EFI not found: crt0-efi-$(ARCH).o missing.)
NO_UEFI = 1
endif
ifeq ($(strip $(EFI_LIBDIR)),)
EFI_LIBDIR := /usr/lib
endif

# UEFI build flags
CFLAGS = -ffreestanding -fno-stack-protector -fpic -fshort-wchar -mno-red-zone \
	 -I/usr/include/efi -I/usr/include/efi/$(ARCH) -DEFI_FUNCTION_WRAPPER \
	 -O2 -msse2

BUILD_ID ?= $(shell date -u +%Y-%m-%dT%H:%M:%SZ)
BUILD_ID := $(BUILD_ID)
CFLAGS += -DLLMB_BUILD_ID=L\"$(BUILD_ID)\"

LDFLAGS = -nostdlib -znocombreloc -T $(EFI_LDS) \
	  -shared -Bsymbolic -L$(EFI_LIBDIR) $(EFI_CRT0)

LIBS = -lefi -lgnuefi

# ── Target ──────────────────────────────────────────────────────────────────

TARGET = llama2.efi
REPL_SRC = llama2_efi_mamba.c
REPL_OBJ = llama2_repl.o

# ── OO Engine Objects ───────────────────────────────────────────────────────

DJIBION_OBJ      = djibion-engine/core/djibion.o
DIOPION_OBJ      = diopion-engine/core/diopion.o
DIAGNOSTION_OBJ  = diagnostion-engine/core/diagnostion.o
MEMORION_OBJ     = memorion-engine/core/memorion.o
ORCHESTRION_OBJ  = orchestrion-engine/core/orchestrion.o
CALIBRION_OBJ    = calibrion-engine/core/calibrion.o
COMPATIBILION_OBJ = compatibilion-engine/core/compatibilion.o
EVOLVION_OBJ     = evolvion-engine/core/evolvion.o
SYNAPTION_OBJ    = synaption-engine/core/synaption.o
CONSCIENCE_OBJ   = conscience-engine/core/conscience.o
NEURALFS_OBJ     = neuralfs-engine/core/neuralfs.o
GHOST_OBJ        = ghost-engine/core/ghost.o
IMMUNION_OBJ     = immunion-engine/core/immunion.o
DREAMION_OBJ     = dreamion-engine/core/dreamion.o
SYMBION_OBJ      = symbion-engine/core/symbion.o
COLLECTIVION_OBJ = collectivion-engine/core/collectivion.o
METABION_OBJ     = metabion-engine/core/metabion.o
CELLION_OBJ      = cellion-engine/core/cellion.o
MORPHION_OBJ     = morphion-engine/core/morphion.o
PHEROMION_OBJ    = pheromion-engine/core/pheromion.o

ENGINE_OBJS = $(DJIBION_OBJ) $(DIOPION_OBJ) $(DIAGNOSTION_OBJ) \
	$(MEMORION_OBJ) $(ORCHESTRION_OBJ) $(CALIBRION_OBJ) \
	$(COMPATIBILION_OBJ) $(EVOLVION_OBJ) $(SYNAPTION_OBJ) \
	$(CONSCIENCE_OBJ) $(NEURALFS_OBJ) $(GHOST_OBJ) \
	$(IMMUNION_OBJ) $(DREAMION_OBJ) $(SYMBION_OBJ) \
	$(COLLECTIVION_OBJ) $(METABION_OBJ) $(CELLION_OBJ) \
	$(MORPHION_OBJ) $(PHEROMION_OBJ)

# ── SSM Engine Objects ──────────────────────────────────────────────────────

SSM_OBJS = ssm_infer.o ssm_weights.o ssm_infer_avx2.o bpe_tokenizer.o oo_handoff.o

# ── All Objects ─────────────────────────────────────────────────────────────

KERNEL_OBJS = llmk_zones.o llmk_log.o llmk_sentinel.o llmk_oo.o
BLAS_OBJS   = djiblas.o djiblas_avx2.o attention_avx2.o
GGUF_OBJS   = gguf_loader.o gguf_infer.o

REPL_OBJS = $(REPL_OBJ) $(ENGINE_OBJS) $(SSM_OBJS) \
	$(KERNEL_OBJS) $(BLAS_OBJS) $(GGUF_OBJS)

REPL_SO = llama2_repl.so

# ── QEMU / OVMF Paths ──────────────────────────────────────────────────────

OVMF_CODE := $(firstword $(wildcard /usr/share/OVMF/OVMF_CODE.fd /usr/share/OVMF/OVMF_CODE_4M.fd /usr/share/qemu/OVMF.fd))
DISK_IMG  := boot.img
DISK_SIZE := 512

# ── Build Rules ─────────────────────────────────────────────────────────────

.PHONY: all clean rebuild python-test export-test qemu qemu-model qemu-nokvm disk disk-model export export-int8 tokenizer data

ifndef NO_UEFI
all: $(TARGET)
	@echo ""
	@echo "════════════════════════════════════════════════════════"
	@echo "  ✅ Mamba2 SSM + RLF Bare-Metal Build Complete"
	@echo "════════════════════════════════════════════════════════"
	@ls -lh $(TARGET)
	@echo ""
else
all:
	@echo "UEFI build skipped (gnu-efi not installed)"
	@echo "Install with: sudo apt install gnu-efi"
	@echo "Python tests still available: make python-test"
endif

# ── REPL (main entry point) ─────────────────────────────────────────────────

$(REPL_OBJ): $(REPL_SRC) djiblas.h interface.h ssm_infer.h ssm_weights.h oo_handoff.h
	$(CC) $(CFLAGS) -c $(REPL_SRC) -o $(REPL_OBJ)

# ── SSM Engine ──────────────────────────────────────────────────────────────

ssm_infer.o: ssm_infer.c ssm_infer.h
	$(CC) $(CFLAGS) -c ssm_infer.c -o ssm_infer.o

ssm_weights.o: ssm_weights.c ssm_weights.h ssm_infer.h
	$(CC) $(CFLAGS) -c ssm_weights.c -o ssm_weights.o

ssm_infer_avx2.o: ssm_infer_avx2.c ssm_infer.h
	$(CC) $(CFLAGS) -mavx2 -mfma -c ssm_infer_avx2.c -o ssm_infer_avx2.o

bpe_tokenizer.o: bpe_tokenizer.c bpe_tokenizer.h
	$(CC) $(CFLAGS) -c bpe_tokenizer.c -o bpe_tokenizer.o

oo_handoff.o: oo_handoff.c oo_handoff.h ssm_infer.h
	$(CC) $(CFLAGS) -c oo_handoff.c -o oo_handoff.o

# ── Kernel Primitives ───────────────────────────────────────────────────────

llmk_zones.o: llmk_zones.c llmk_zones.h
	$(CC) $(CFLAGS) -c llmk_zones.c -o llmk_zones.o

llmk_log.o: llmk_log.c llmk_log.h llmk_zones.h
	$(CC) $(CFLAGS) -c llmk_log.c -o llmk_log.o

llmk_sentinel.o: llmk_sentinel.c llmk_sentinel.h llmk_zones.h llmk_log.h
	$(CC) $(CFLAGS) -c llmk_sentinel.c -o llmk_sentinel.o

llmk_oo.o: llmk_oo.c llmk_oo.h
	$(CC) $(CFLAGS) -c llmk_oo.c -o llmk_oo.o

# ── BLAS ────────────────────────────────────────────────────────────────────

djiblas.o: djiblas.c djiblas.h
	$(CC) $(CFLAGS) -c djiblas.c -o djiblas.o

djiblas_avx2.o: djiblas_avx2.c djiblas.h
	$(CC) $(CFLAGS) -mavx2 -mfma -c djiblas_avx2.c -o djiblas_avx2.o

attention_avx2.o: attention_avx2.c
	$(CC) $(CFLAGS) -mavx2 -mfma -c attention_avx2.c -o attention_avx2.o

# ── GGUF ────────────────────────────────────────────────────────────────────

gguf_loader.o: gguf_loader.c gguf_loader.h
	$(CC) $(CFLAGS) -c gguf_loader.c -o gguf_loader.o

gguf_infer.o: gguf_infer.c gguf_infer.h
	$(CC) $(CFLAGS) -c gguf_infer.c -o gguf_infer.o

# ── OO Engines ──────────────────────────────────────────────────────────────

djibion-engine/core/djibion.o: djibion-engine/core/djibion.c djibion-engine/core/djibion.h
	$(CC) $(CFLAGS) -c $< -o $@

diopion-engine/core/diopion.o: diopion-engine/core/diopion.c diopion-engine/core/diopion.h
	$(CC) $(CFLAGS) -c $< -o $@

diagnostion-engine/core/diagnostion.o: diagnostion-engine/core/diagnostion.c diagnostion-engine/core/diagnostion.h
	$(CC) $(CFLAGS) -c $< -o $@

memorion-engine/core/memorion.o: memorion-engine/core/memorion.c memorion-engine/core/memorion.h
	$(CC) $(CFLAGS) -c $< -o $@

orchestrion-engine/core/orchestrion.o: orchestrion-engine/core/orchestrion.c orchestrion-engine/core/orchestrion.h
	$(CC) $(CFLAGS) -c $< -o $@

calibrion-engine/core/calibrion.o: calibrion-engine/core/calibrion.c calibrion-engine/core/calibrion.h
	$(CC) $(CFLAGS) -c $< -o $@

compatibilion-engine/core/compatibilion.o: compatibilion-engine/core/compatibilion.c compatibilion-engine/core/compatibilion.h
	$(CC) $(CFLAGS) -c $< -o $@

evolvion-engine/core/evolvion.o: evolvion-engine/core/evolvion.c evolvion-engine/core/evolvion.h
	$(CC) $(CFLAGS) -c $< -o $@

synaption-engine/core/synaption.o: synaption-engine/core/synaption.c synaption-engine/core/synaption.h
	$(CC) $(CFLAGS) -c $< -o $@

conscience-engine/core/conscience.o: conscience-engine/core/conscience.c conscience-engine/core/conscience.h
	$(CC) $(CFLAGS) -c $< -o $@

neuralfs-engine/core/neuralfs.o: neuralfs-engine/core/neuralfs.c neuralfs-engine/core/neuralfs.h
	$(CC) $(CFLAGS) -c $< -o $@

ghost-engine/core/ghost.o: ghost-engine/core/ghost.c ghost-engine/core/ghost.h
	$(CC) $(CFLAGS) -c $< -o $@

immunion-engine/core/immunion.o: immunion-engine/core/immunion.c immunion-engine/core/immunion.h
	$(CC) $(CFLAGS) -c $< -o $@

dreamion-engine/core/dreamion.o: dreamion-engine/core/dreamion.c dreamion-engine/core/dreamion.h
	$(CC) $(CFLAGS) -c $< -o $@

symbion-engine/core/symbion.o: symbion-engine/core/symbion.c symbion-engine/core/symbion.h
	$(CC) $(CFLAGS) -c $< -o $@

collectivion-engine/core/collectivion.o: collectivion-engine/core/collectivion.c collectivion-engine/core/collectivion.h
	$(CC) $(CFLAGS) -c $< -o $@

metabion-engine/core/metabion.o: metabion-engine/core/metabion.c metabion-engine/core/metabion.h
	$(CC) $(CFLAGS) -c $< -o $@

cellion-engine/core/cellion.o: cellion-engine/core/cellion.c cellion-engine/core/cellion.h
	$(CC) $(CFLAGS) -c $< -o $@

morphion-engine/core/morphion.o: morphion-engine/core/morphion.c morphion-engine/core/morphion.h
	$(CC) $(CFLAGS) -c $< -o $@

pheromion-engine/core/pheromion.o: pheromion-engine/core/pheromion.c pheromion-engine/core/pheromion.h
	$(CC) $(CFLAGS) -c $< -o $@

# ── Link + EFI ──────────────────────────────────────────────────────────────

ifndef NO_UEFI
$(REPL_SO): $(REPL_OBJS)
	ld $(LDFLAGS) $(REPL_OBJS) -o $(REPL_SO) $(LIBS)

$(TARGET): $(REPL_SO)
	objcopy -j .text -j .sdata -j .data -j .dynamic -j .dynsym \
		-j .rel -j .rela -j .reloc --target=efi-app-$(ARCH) $(REPL_SO) $(TARGET)
endif

# ── QEMU Testing ────────────────────────────────────────────────────────────

disk: $(TARGET)
	@echo "Creating $(DISK_SIZE)MB FAT32 boot image..."
	dd if=/dev/zero of=$(DISK_IMG) bs=1M count=$(DISK_SIZE) 2>/dev/null
	mkfs.fat -F 32 $(DISK_IMG)
	mmd -i $(DISK_IMG) ::/EFI
	mmd -i $(DISK_IMG) ::/EFI/BOOT
	mcopy -i $(DISK_IMG) $(TARGET) ::/EFI/BOOT/BOOTX64.EFI
	@echo "  ✅ Created $(DISK_IMG)"

disk-model: disk
	@test -f model.mamba.bin && mcopy -i $(DISK_IMG) model.mamba.bin ::/model.mamba.bin && echo "  ✅ Added model" || echo "  ⚠ No model.mamba.bin"
	@test -f tokenizer.bpe.bin && mcopy -i $(DISK_IMG) tokenizer.bpe.bin ::/tokenizer.bpe.bin && echo "  ✅ Added tokenizer" || true
	@test -f OOHANDOFF.TXT && mcopy -i $(DISK_IMG) OOHANDOFF.TXT ::/OOHANDOFF.TXT && echo "  ✅ Added handoff receipt" || true

qemu: disk
	qemu-system-x86_64 -bios $(OVMF_CODE) -drive file=$(DISK_IMG),format=raw \
		-m 2G -cpu host,+avx2 -enable-kvm -nographic -serial mon:stdio

qemu-model: disk-model
	qemu-system-x86_64 -bios $(OVMF_CODE) -drive file=$(DISK_IMG),format=raw \
		-m 4G -cpu host,+avx2 -enable-kvm -nographic -serial mon:stdio

qemu-nokvm: disk
	qemu-system-x86_64 -bios $(OVMF_CODE) -drive file=$(DISK_IMG),format=raw \
		-m 2G -cpu qemu64,+avx2,+fma -nographic -serial mon:stdio

# ── Python Tools ────────────────────────────────────────────────────────────

python-test:
	python quick_test.py

export-test:
	python export_mamba_baremetal.py --test model_test.mamba.bin

export:
	python export_mamba_baremetal.py

export-int8:
	python export_mamba_baremetal.py --quantize int8

tokenizer:
	python export_bpe_table.py

data:
	python data_builder_v2.py

# ── Clean ───────────────────────────────────────────────────────────────────

clean:
	rm -f $(REPL_OBJS) $(REPL_SO) $(TARGET) $(DISK_IMG)
	rm -f ssm_infer.o ssm_weights.o ssm_infer_avx2.o bpe_tokenizer.o oo_handoff.o
	rm -f *.mamba.bin
	@echo ""
	@echo "✅ Clean complete"

rebuild: clean all

