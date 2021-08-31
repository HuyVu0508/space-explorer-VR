using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class floating_updown : MonoBehaviour
{
    public float frequency = 1f;
    public float amplitude = 0.5f;
    private Vector3 original_position;
    private Vector3 temporary_position;

    // Start is called before the first frame update
    void Start()
    {
        // original_position = transform.position;
    }

    // Update is called once per frame
    void Update()
    {
        // temporary_position = original_position;
        // temporary_position.y += Mathf.Sin (Time.fixedTime * Mathf.PI * frequency) * amplitude;
        // transform.position = temporary_position;
    }

}