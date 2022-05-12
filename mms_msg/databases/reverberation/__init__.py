"""
An example in a RIR database should have the following structure:

```python
{
    'audio_path': {
        'rir': [
            'path/to/rir1.wav',
            'path/to/rir2.wav',
            ...
        ]
    },
    # Optional
    num_samples: {
        'rir': [
            42, 42, ...
        ]
    }
}
```
"""
