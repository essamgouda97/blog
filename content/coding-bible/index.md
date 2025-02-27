---
title: 'Coding Bible ✝️'
date: '2025-02-19'
excerpt: 'This article will be updated as I find more rules that helped me in my coding journey.'
---

This article will be updated as I find more rules that helped me in my coding journey. Below are the coding practices and principles that I follow.

## 1. Hooks = readable code

Good hooks lead to clean and readable code.

## 2. Code the task immediately, debug later

Start coding the task without worrying too much about perfection. Debugging comes afterward.

## 3. Tests, Tests, Tests

Testing is vital for maintaining code quality and ensuring stability.

## 4. Proper debugging skills are a must to understand any codebase

A solid understanding of debugging will help you comprehend and fix issues within any codebase.

## 5. Define happy paths in your flow

Ensure you have a clear, happy path defined in your code flow for expected behavior.

## 6. New work rules

   1. Meet as many people as possible. Ask them what they do, what issues they are facing (technical and non-technical), and what they plan to work on next.
   2. Understand the organizational chart.
   3. Draw an end-to-end flow for everything you plan to touch.
   4. Do an end-to-end run with a debugger.

## 7. Sketch high-level, then iterate between low-level designs and coding

Start with high-level designs and then move to detailed designs, iterating between design and coding.

## 8. Rollbacks must be smooth

Ensure your rollback processes are simple and smooth to avoid headaches.

## 9. Follow [SemVer](https://semver.org/) `MAJOR.MINOR.PATCH (XX.XX.XX)`

   1. **MAJOR** version when you make incompatible API changes.
   2. **MINOR** version when you add functionality in a backward-compatible manner.
   3. **PATCH** version when you make backward-compatible bug fixes.

## 10. `{FUNCTION-SERVICE}_ENVVARNAME` for env vars e.g., `REDIS_HOST`.

Always use clear and consistent naming for environment variables, following a predictable structure.

## 11. Low level whenever you can. The less abstractions (especially initially) the better

Avoid excessive abstractions in the beginning. Focus on simplicity and clarity.

## 12. Instead of doing destructive actions directly, implement a simulation mode

If the simulation mode is active, log the action in an SQLite database instead of deleting anything.

## 13. Dependency injection. Functional programming. (Take in progress)

Make use of dependency injection and incorporate functional programming principles where possible.

## 14. CONSISTENT NAMING CONVENTIONS FOR LIFE!

Stick to a consistent naming convention for all your variables, functions, and services.

## 15. Delays aren’t always a bad thing, especially with many requests to a service

In cases of multiple requests to a service, delays might actually help with load balancing and reliability.

## 16. Less assumptions, more debugging

Don't assume what's wrong. Debug it.

## 17. No one really likes `yamls`

YAMLs can get messy quickly. Use them with care and avoid unnecessary complexity.

## 18. Never clean up at once (unless it's a quick cleanup, of course)

Start cleaning up iteratively, piece by piece, to avoid large-scale disruptions.

## 19. Understand the git graph

You need to be able to navigate commits like a time traveler:  
`git log --reverse --oneline <starting-commit-hash>..main`
