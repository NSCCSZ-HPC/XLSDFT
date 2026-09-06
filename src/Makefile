include Makefile.config

include objects.mk

# make 会自动选择 Makefile 文件中出现的第一个目标。
all: a.out

.PHONY: objs
objs: $(OBJSC)

%.o : %.cpp
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $@

a.out: $(OBJSC)
	$(CC) -o $(BINBASE) $^ $(CFLAGS) $(LDFLAGS) $(LDLIBS)

.PHONY: clean
clean:
	rm -f $(OBJSC) $(BINBASE)
