# compiler settings
CXX = nvcc
CXXFLAGS_debug = -g -G
CXXFLAGS_release = -O3

# build files
COMMON_CPP_FILES = CpuUpscaler.cpp Utility.cpp
COMMON_CU_FILES = GpuUpscaler.cu
MAIN_FILE = Main.cpp
PROFILE_MAIN_FILE = ProfileMain.cpp
TEST_FILE = Test.cpp

# output directory
SOURCE_DIR = src
BUILD_DIR  = bin

# executable setup
EXECUTABLE = upscaler
EXECUTABLE_PATH = $(BUILD_DIR)/$(EXECUTABLE)

# compilation mode setup
ifeq ($(mode), debug)
	CXXFLAGS = $(CXXFLAGS_debug)
else
	CXXFLAGS = $(CXXFLAGS_release)
	mode = release
endif

# main file setup
ifeq ($(main), main)
	CHOSEN_MAIN_FILE = $(MAIN_FILE)
else ifeq ($(main), profile)
	CHOSEN_MAIN_FILE = $(PROFILE_MAIN_FILE)
	main = profile
else
	CHOSEN_MAIN_FILE = $(TEST_FILE)
	main = test
endif

# common object list
COMMON_CPP_OBJECTS = $(addprefix $(BUILD_DIR)/, $(COMMON_CPP_FILES:.cpp=.o)) 
COMMON_CU_OBJECTS = $(addprefix $(BUILD_DIR)/, $(COMMON_CU_FILES:.cu=.o))

# main object file
MAIN_OBJECT = $(BUILD_DIR)/$(CHOSEN_MAIN_FILE:.cpp=.o)

.PHONY: all clean

# defined target
all: $(EXECUTABLE_PATH)

# main object compile rule
$(MAIN_OBJECT): $(SOURCE_DIR)/$(CHOSEN_MAIN_FILE)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# cpp common objects compile rule
$(BUILD_DIR)/%.o: $(SOURCE_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# cu common objects compile rule
$(BUILD_DIR)/%.o: $(SOURCE_DIR)/%.cu
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# executable creation rule
$(EXECUTABLE_PATH): $(COMMON_CPP_OBJECTS) $(COMMON_CU_OBJECTS) $(MAIN_OBJECT)
	$(CXX) $^ -o $@

# clean rule
clean:
	rm -rf $(BUILD_DIR)

# help command
help:
	@echo "Usage: make [mode=(debug/release)] [main=(main/profile/test)]"
	@echo "Default: make (compile with Test.cpp in Release mode)"
	@echo "Options:"
	@echo "  mode=debug/release        	- Choose the compile mode"
	@echo "  main=main/profile/test    	- Choose the main file"
