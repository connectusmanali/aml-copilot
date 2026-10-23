# Layering: cycles and pass-through

## Definition
Layering moves funds through several accounts to hide their origin. Two common shapes: a cycle, where money leaves an account and returns through intermediaries (A to B to C to A), and pass-through, where an account receives funds and sends most of them on within a day.

## Red flags
Money returning to its origin within days, often slightly reduced at each hop. Received and sent amounts within 24 hours that nearly match. Accounts with little activity of their own between the transfers.

## Investigation steps
Trace the subject's counterparties and look for paths that return to the subject in time order. Compare inflows and outflows over 24 hours. Note any fee-like reduction at each hop.

## Common false positives
A refund: A pays B and B returns the same amount later. A refund is two accounts, not three, and usually carries a merchant relationship.
