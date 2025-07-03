#!/bin/bash

PACKAGE_NAME="mindspore"
FUNC_NAME="acldumpRegCallback"
RELATIVE_PATH="lib/plugin/libmindspore_ascend.so.2"

PACKAGE_INFO=$(pip show "$PACKAGE_NAME")

if [ $? -ne 0 ]; then
    echo "Package '$PACKAGE_NAME' not found."
    exit 1
fi

LOCATION=$(echo "$PACKAGE_INFO" | grep '^Location:' | awk '{print $2}')

if [ -z "$LOCATION" ]; then
    echo "Could not find the location for package '$PACKAGE_NAME'."
    exit 1
fi

SO_FILE="$LOCATION/$PACKAGE_NAME/$RELATIVE_PATH"

if [ ! -f "$SO_FILE" ]; then
    echo "SO file '$SO_FILE' does not exist."
    exit 1
fi

SYMBOL_INFO=$(nm -D -C "$SO_FILE" | grep $FUNC_NAME)

if [ -z "$SYMBOL_INFO" ]; then
    echo "Symbol '$FUNC_NAME' not found in '$SO_FILE'."
    exit 1
else
    SYMBOL_TYPE=$(echo "$SYMBOL_INFO" | awk '{print $2}')
    if [ "$SYMBOL_TYPE" == "T" ]; then
        echo "Successful, symbol '$FUNC_NAME' is a Global Function (Type T) in '$SO_FILE'."
    else
        echo "Symbol '$FUNC_NAME' found in '$SO_FILE', but its type is '$SYMBOL_TYPE', not 'T'."
        exit 1
    fi
fi
