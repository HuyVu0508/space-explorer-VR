using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class cutting_trees : MonoBehaviour
{
    public GameObject scissor;
    public float threshold_distance = 3;

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        Vector3 distance_vector = scissor.transform.position - transform.position;
        float distance = distance_vector.magnitude;

        if (distance < threshold_distance){
            transform.gameObject.SetActive(false);
        }        
    }
}
