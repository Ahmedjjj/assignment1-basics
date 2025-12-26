# `unicode1`
## a
`\x00`
## b
Printing this character does not display anything.
## c 
It is ignored (replaced by '').
# `unicode2`
## a
UTF-8 tends to produce shorter byte sequences for common characters as it uses variable width encoding.
## b
input: "الكرام" throws an error.
Function assumes each bytes decodes to 1 character. But utf-8 uses more than 1 byte for non-ascii characters
## c
\x80\x11
\x80 is a continuation byte, a valid utf-8 string cannot start with it