rule rule1:
    input: "test.h5::1"
    output: "test.h5::2"
    run:
        pass

rule All:
    input: "test.h5::2"
